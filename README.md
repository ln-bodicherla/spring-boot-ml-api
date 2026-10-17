# Spring Boot ML API

ML model serving microservice built with Spring Boot (Java). Provides REST endpoints for fraud detection model inference with health checks, circuit breakers (Resilience4j), metrics (Micrometer/Prometheus), and containerized deployment.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        Spring Boot ML API        │
                    │                                  │
    HTTP Request    │  ┌──────────────────────────┐   │
   ─────────────────┤  │   PredictionController    │   │
                    │  │  - /api/v1/predict        │   │
                    │  │  - /api/v1/predict/batch  │   │
                    │  │  - /api/v1/model/info     │   │
                    │  └────────────┬───────────────┘   │
                    │               │                   │
                    │  ┌────────────▼───────────────┐   │
                    │  │   PreprocessingService     │   │
                    │  │  - Feature normalization   │   │
                    │  │  - Input validation        │   │
                    │  │  - Categorical encoding    │   │
                    │  └────────────┬───────────────┘   │
                    │               │                   │
                    │  ┌────────────▼───────────────┐   │
                    │  │      ModelService          │   │
                    │  │  - ONNX Runtime inference  │   │
                    │  │  - Circuit breaker         │   │
                    │  │  - Model warmup            │   │
                    │  └────────────┬───────────────┘   │
                    │               │                   │
                    │  ┌────────────▼───────────────┐   │
                    │  │    ONNX Runtime Engine     │   │
                    │  │  - Optimized inference     │   │
                    │  │  - Multi-threaded          │   │
                    │  └───────────────────────────┘   │
                    │                                  │
                    │  ┌───────────────────────────┐   │
                    │  │     Actuator Endpoints     │   │
                    │  │  - /actuator/health        │   │
                    │  │  - /actuator/metrics       │   │
                    │  │  - /actuator/prometheus    │   │
                    │  └───────────────────────────┘   │
                    └─────────────────────────────────┘
```

## Why Spring Boot Over FastAPI

This project explores Spring Boot as an alternative to FastAPI for ML model serving:

| Aspect | Spring Boot (Java) | FastAPI (Python) |
|--------|-------------------|------------------|
| Language | Java 17+ | Python 3.10+ |
| Ecosystem | Enterprise Java (massive) | Python ML (native) |
| Type Safety | Compile-time | Runtime (Pydantic) |
| Startup Time | ~3-5 seconds | ~0.5-1 second |
| Inference Overhead | ONNX Runtime Java | Native PyTorch/TF |
| Serialization | Jackson (fast) | Pydantic (flexible) |
| Monitoring | Micrometer/Actuator | Custom/Prometheus client |
| Circuit Breakers | Resilience4j (native) | Custom implementation |

**Decision**: FastAPI (see `fastapi-ml-microservice`) was chosen for the primary serving project due to tighter integration with the Python ML ecosystem (PyTorch, scikit-learn), lower serialization overhead when working with NumPy arrays, and faster iteration during model development.

## Project Structure

```
spring-boot-ml-api/
├── src/main/java/com/portfolio/mlapi/
│   ├── MlApiApplication.java          # Application entry point
│   ├── controller/
│   │   └── PredictionController.java   # REST endpoints
│   ├── service/
│   │   ├── ModelService.java           # ONNX inference + circuit breaker
│   │   └── PreprocessingService.java   # Feature preprocessing
│   ├── model/
│   │   └── PredictionRequest.java      # DTOs and validation
│   └── config/
│       └── ModelConfig.java            # ONNX Runtime configuration
├── src/main/resources/
│   └── application.yml                 # Spring configuration
├── docker/
│   └── Dockerfile                      # Container image
├── pom.xml                             # Maven dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Java 17+
- Maven 3.8+
- ONNX model file (see below for generating one)

### Build and Run

```bash
# Build
mvn clean package -DskipTests

# Run
java -jar target/ml-api-1.0.0.jar

# Or with Maven
mvn spring-boot:run
```

### Docker

```bash
# Build image
docker build -f docker/Dockerfile -t ml-api:latest .

# Run container
docker run -p 8080:8080 ml-api:latest
```

## API Reference

### Predict

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, -1.2, 0.3, 2.1, -0.8, 0.1, 1.5, -0.3, 0.7, -1.1],
    "featureNames": ["amount", "time", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]
  }'
```

### Batch Predict

```bash
curl -X POST http://localhost:8080/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"features": [0.5, -1.2, 0.3, 2.1, -0.8, 0.1, 1.5, -0.3, 0.7, -1.1]},
      {"features": [1.0, 0.5, -0.3, 0.8, 1.2, -0.5, 0.3, 0.9, -0.1, 0.4]}
    ]
  }'
```

### Model Info

```bash
curl http://localhost:8080/api/v1/model/info
```

### Health Check

```bash
curl http://localhost:8080/api/v1/health
```

## Monitoring

### Prometheus Metrics

```bash
curl http://localhost:8080/actuator/prometheus
```

Key metrics:
- `prediction_requests_total` - Total prediction requests
- `prediction_latency_seconds` - Prediction latency histogram
- `model_inference_errors_total` - Inference error count
- `circuit_breaker_state` - Circuit breaker status

### Health Endpoint

```bash
curl http://localhost:8080/actuator/health
```

## Configuration

See `src/main/resources/application.yml` for all configurable properties:

- Model path, version, and warmup settings
- Server port and thread pool
- Circuit breaker thresholds
- Prometheus metrics export
- Actuator endpoint exposure

## Tech Stack

- **Framework**: Spring Boot 3.x
- **Language**: Java 17
- **Inference**: ONNX Runtime
- **Resilience**: Resilience4j
- **Metrics**: Micrometer + Prometheus
- **Build**: Maven
- **Container**: Docker (Eclipse Temurin JRE)
