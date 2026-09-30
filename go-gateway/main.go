package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	pb "go-gateway/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ChatCompletionRequest defines the JSON payload received from HTTP clients.
type ChatCompletionRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int32   `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
}

// StreamChunk represents the Server-Sent Event (SSE) JSON structure.
type StreamChunk struct {
	Token   string `json:"token"`
	IsFinal bool   `json:"is_final"`
}

// GatewayServer holds our gRPC client and HTTP routing handlers.
type GatewayServer struct {
	grpcClient pb.InferenceServiceClient
}

func main() {
	grpcWorkerAddr := os.Getenv("GRPC_WORKER_ADDR")
	if grpcWorkerAddr == "" {
		grpcWorkerAddr = "127.0.0.1:50051"
	}

	httpPort := os.Getenv("PORT")
	if httpPort == "" {
		httpPort = "8080"
	}

	log.Printf("Connecting to Rust Gemma 4 gRPC worker at %s...", grpcWorkerAddr)

	// Establish persistent gRPC connection to the Rust backend
	conn, err := grpc.NewClient(
		grpcWorkerAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("Failed to connect to gRPC worker: %v", err)
	}
	defer conn.Close()

	client := pb.NewInferenceServiceClient(conn)
	server := &GatewayServer{grpcClient: client}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", server.handleChatCompletions)
	mux.HandleFunc("/health", server.handleHealth)
	mux.HandleFunc("/", server.handleRoot)

	httpServer := &http.Server{
		Addr:         ":" + httpPort,
		Handler:      corsMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // 0 allows unbounded streaming for Server-Sent Events (SSE)
	}

	// Graceful shutdown handling
	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("Go API Gateway listening on http://0.0.0.0:%s", httpPort)
		log.Printf("Endpoint available: POST http://localhost:%s/v1/chat/completions", httpPort)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	<-stopChan
	log.Println("Shutting down Go API Gateway gracefully...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("Error during server shutdown: %v", err)
	}
	log.Println("Gateway exited.")
}

// handleChatCompletions handles incoming requests and streams tokens over Server-Sent Events (SSE).
func (s *GatewayServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed. Use POST.", http.StatusMethodNotAllowed)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported by response writer", http.StatusInternalServerError)
		return
	}

	// Parse JSON request
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "Field 'prompt' cannot be empty", http.StatusBadRequest)
		return
	}

	if req.MaxTokens <= 0 {
		req.MaxTokens = 128
	}

	log.Printf("Incoming prompt: %q (max_tokens=%d, temp=%.2f)", req.Prompt, req.MaxTokens, req.Temperature)

	// Set Server-Sent Events headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher.Flush()

	// Call Rust gRPC StreamGenerate
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	stream, err := s.grpcClient.StreamGenerate(ctx, &pb.GenerateRequest{
		Prompt:      req.Prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	})
	if err != nil {
		log.Printf("gRPC StreamGenerate call failed: %v", err)
		chunkBytes, _ := json.Marshal(StreamChunk{Token: fmt.Sprintf("\n[Error: %v]", err), IsFinal: true})
		fmt.Fprintf(w, "data: %s\n\n", chunkBytes)
		flusher.Flush()
		return
	}

	// Stream tokens as they arrive from Rust worker
	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Printf("Stream interrupted: %v", err)
			break
		}

		chunk := StreamChunk{
			Token:   resp.Token,
			IsFinal: resp.IsFinal,
		}
		chunkBytes, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", chunkBytes)
		flusher.Flush()

		if resp.IsFinal {
			break
		}
	}

	// Send standard SSE end-of-stream delimiter
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
	log.Printf("Stream completed successfully.")
}

// handleHealth returns a simple 200 OK health check.
func (s *GatewayServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
		"engine": "Gemma 4 Rust + Candle Worker",
	})
}

// handleRoot displays a friendly landing message and usage instructions.
func (s *GatewayServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, `
		<html>
		<head><title>Gemma 4 High-Concurrency API Gateway</title></head>
		<body style="font-family: sans-serif; padding: 40px; background: #18181b; color: #f4f4f5;">
			<h1>Google Gemma 4 High-Concurrency API Gateway</h1>
			<p>Powered by Go HTTP/2 Gateway + Rust Candle gRPC Engine.</p>
			<h3>Send a streaming chat completion:</h3>
			<pre style="background: #27272a; padding: 16px; border-radius: 8px; color: #a1a1aa;">
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain why Rust and Go make a great pair.", "max_tokens": 100}'
			</pre>
		</body>
		</html>
	`)
}

// corsMiddleware adds standard CORS headers for web client integrations.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}
