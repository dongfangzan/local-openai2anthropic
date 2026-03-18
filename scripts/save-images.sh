#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Save Docker images to tar files for offline use

set -e

OUTPUT_DIR="${OUTPUT_DIR:-./docker-images}"
VERSION="${VERSION:-latest}"

echo "Saving Docker images to $OUTPUT_DIR..."
mkdir -p "$OUTPUT_DIR"

# Build images first
echo "Building images..."
docker-compose -f docker-compose.build.yml build

# Save OA2A image
echo "Saving OA2A proxy image..."
docker save local-openai2anthropic-oa2a:latest > "$OUTPUT_DIR/oa2a-$VERSION.tar"

# Save Claude Code image
echo "Saving Claude Code image..."
docker save local-openai2anthropic-claude-code:latest > "$OUTPUT_DIR/claude-code-$VERSION.tar"

# Create load script
cat > "$OUTPUT_DIR/load-images.sh" << 'EOF'
#!/bin/bash
# Load Docker images from tar files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Loading Docker images..."

docker load < "$SCRIPT_DIR/oa2a-latest.tar"
docker load < "$SCRIPT_DIR/claude-code-latest.tar"

echo "Images loaded successfully!"
echo ""
echo "Tagged as:"
echo "  - dongfangzan/local-openai2anthropic:latest"
echo "  - dongfangzan/claude-code:latest"
EOF

chmod +x "$OUTPUT_DIR/load-images.sh"

echo ""
echo "Images saved to $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "To load images on another machine:"
echo "  cd $OUTPUT_DIR && ./load-images.sh"
