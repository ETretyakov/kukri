package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	hugot "github.com/knights-analytics/hugot"
)

func main() {
	model := flag.String("model", "KnightsAnalytics/all-MiniLM-L6-v2", "HuggingFace model name")
	dest := flag.String("dest", "./models", "destination directory")
	flag.Parse()

	ctx := context.Background()

	opts := hugot.NewDownloadOptions()
	opts.Verbose = true

	fmt.Fprintf(os.Stderr, "downloading %s → %s\n", *model, *dest)

	path, err := hugot.DownloadModel(ctx, *model, *dest, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(path)
}
