FROM docker.io/library/golang:1.26.5-alpine@sha256:0178a641fbb4858c5f1b48e34bdaabe0350a330a1b1149aabd498d0699ff5fb2 AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download && go mod verify
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /gpu-integrity-watch .

FROM docker.io/library/alpine:3.23@sha256:fd791d74b68913cbb027c6546007b3f0d3bc45125f797758156952bc2d6daf40
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
RUN mkdir -p /var/lib/secure-ai/integrity /var/lib/secure-ai/logs && \
    chown -R 65534:65534 /var/lib/secure-ai
COPY --from=build /gpu-integrity-watch /usr/local/bin/gpu-integrity-watch
COPY profiles/default-profile.yaml /etc/secure-ai/gpu-integrity-profile.yaml
ENV INTEGRITY_PROFILE=/etc/secure-ai/gpu-integrity-profile.yaml
USER 65534:65534
EXPOSE 8505
VOLUME ["/var/lib/secure-ai"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 CMD wget -q -T 3 -O - http://127.0.0.1:8505/health >/dev/null || exit 1
ENTRYPOINT ["gpu-integrity-watch"]
CMD ["daemon"]
