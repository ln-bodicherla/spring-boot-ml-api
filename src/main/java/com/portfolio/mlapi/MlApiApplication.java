package com.portfolio.mlapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot ML API application entry point.
 *
 * Serves an ONNX fraud detection model via REST endpoints with
 * circuit breaker fault tolerance, Prometheus metrics, and health monitoring.
 */
@SpringBootApplication
public class MlApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(MlApiApplication.class, args);
    }
}
