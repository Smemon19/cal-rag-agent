
# Introduction to Token Buckets

The Token Bucket algorithm is commonly used for rate-limiting in network applications. It allows for bursts of data as long as tokens are available.

## Core Concepts

- **Bucket Capacity:** The maximum number of tokens the bucket can hold.
- **Token Generation Rate:** How often new tokens are added to the bucket.
- **Token Consumption:** Each request consumes a token. If no tokens are available, the request is delayed or dropped.

## Example

Imagine a bucket that fills with 1 token per second up to a capacity of 10. If 5 requests come in at once, they all pass. If another 6th request comes in immediately, it has to wait.

## Python Implementation

```python
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_check = time.time()

    def allow_request(self):
        now = time.time()
        elapsed = now - self.last_check
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_check = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

This document is meant to test your RAG system with a chunkable, code-heavy markdown file.
