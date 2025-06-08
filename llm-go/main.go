package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
)

const (
	PhiModelName = "phi"
	OllamaURL    = "http://localhost:11434/api/generate"
)

type Insight struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}

type InsightRequest struct {
	Text string `json:"text"`
}

type InsightResponse struct {
	Insights []Insight `json:"insights"`
}

type OllamaRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func main() {
	// Parse command line flags
	var port int
	flag.IntVar(&port, "port", 0, "Port to run the server on (default: 8080 or PORT env var)")
	flag.Parse()

	// Check environment variable if flag not provided
	if port == 0 {
		if envPort := os.Getenv("PORT"); envPort != "" {
			var err error
			port, err = strconv.Atoi(envPort)
			if err != nil {
				log.Printf("Invalid PORT environment variable: %v, using default port", err)
				port = 8081
			}
		} else {
			port = 8081
		}
	}

	// Check if Ollama is running
	_, err := http.Get("http://localhost:11434/api/version")
	if err != nil {
		log.Fatalf("Ollama is not running. Please start Ollama first: %v", err)
	}

	// Pull the model if not already available
	log.Println("Making sure phi model is available...")
	pullModel()

	r := gin.Default()

	r.POST("/extract/insights", func(c *gin.Context) {
		var req InsightRequest
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Define system prompt
		systemPrompt := `
			You the world best scientist specializing in human biology and understanding impact of body metrics
			on the person's productivity, ability to accomplish tasks efficiently and plan the day correctly 
			You are given a scientific paper and your goal is to extract key insights from the text. 
			For each insight, provide a numerical ID and the insight text in the following format:
			1: [insight text]
			2: [insight text]
			3: [insight text]
			`

		// Generate response using Ollama
		ollamaReq := OllamaRequest{
			Model:       PhiModelName,
			Prompt:      systemPrompt + req.Text,
			Stream:      false,
			Temperature: 0.7,
			TopP:        0.9,
		}

		reqJSON, _ := json.MarshalIndent(ollamaReq, "", "  ")
		log.Printf("Sending request to LLM:\n%s", string(reqJSON))

		insights, err := callOllama(ollamaReq)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate insights: " + err.Error()})
			return
		}

		parsedInsights := parseInsights(insights)
		c.JSON(http.StatusOK, InsightResponse{
			Insights: parsedInsights,
		})

		fmt.Println(parsedInsights)
	})

	// Try the specified port first
	serverAddr := fmt.Sprintf(":%d", port)
	log.Printf("Server starting on port %d", port)
	err = r.Run(serverAddr)

	// If port is in use, try another port
	if err != nil {
		log.Printf("Failed to start server on port %d: %v", port, err)
		fallbackPort := 0 // Let the system choose an available port
		serverAddr = fmt.Sprintf(":%d", fallbackPort)
		log.Printf("Trying alternative port %d", fallbackPort)
		if err := r.Run(serverAddr); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}
}

func parseInsights(response string) []Insight {
	var insights []Insight
	lines := strings.Split(response, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}

		id, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}

		text := strings.TrimSpace(parts[1])
		if text != "" {
			insights = append(insights, Insight{
				ID:   id,
				Text: text,
			})
		}
	}
	return insights
}

func pullModel() {
	reqBody := map[string]string{
		"name": PhiModelName,
	}

	reqBytes, _ := json.Marshal(reqBody)

	resp, err := http.Post("http://localhost:11434/api/pull", "application/json", bytes.NewBuffer(reqBytes))
	if err != nil {
		log.Printf("Warning: could not pull model: %v", err)
		return
	}
	defer resp.Body.Close()

	log.Println("Model phi2 is ready")
}

func callOllama(req OllamaRequest) (string, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := http.Post(OllamaURL, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to call Ollama: %w", err)
	}
	defer resp.Body.Close()

	var result string

	// Process the response
	decoder := json.NewDecoder(resp.Body)
	for {
		var ollResp OllamaResponse
		if err := decoder.Decode(&ollResp); err != nil {
			if err == io.EOF {
				break
			}
			return "", fmt.Errorf("failed to decode response: %w", err)
		}

		result += ollResp.Response

		if ollResp.Done {
			break
		}
	}

	return result, nil
}
