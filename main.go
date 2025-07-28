package main

import (
	"math/rand"
	"time"

	"go-neuralnetwork/internal/cli"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	cli.RunCLI()
}
