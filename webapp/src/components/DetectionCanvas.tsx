import { useEffect, useRef } from 'react';
import { Detection } from '../types/detection';

interface DetectionCanvasProps {
  imageUrl: string;
  detections: Detection[];
  selectedDetectionId?: string;
}

export default function DetectionCanvas({ imageUrl, detections, selectedDetectionId }: DetectionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    const container = containerRef.current;

    if (!canvas || !image || !container) return;

    const drawDetections = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Wait for image to be loaded and have dimensions
      if (!image.complete || image.naturalWidth === 0) {
        return;
      }

      // Get the actual rendered size of the image element
      const rect = image.getBoundingClientRect();
      const displayedWidth = rect.width;
      const displayedHeight = rect.height;

      if (displayedWidth === 0 || displayedHeight === 0) {
        return;
      }

      // Set canvas internal resolution to match displayed size
      canvas.width = displayedWidth;
      canvas.height = displayedHeight;

      // Set canvas CSS size to match displayed size
      canvas.style.width = `${displayedWidth}px`;
      canvas.style.height = `${displayedHeight}px`;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      console.log('Drawing detections:', {
        count: detections.length,
        displayedWidth,
        displayedHeight,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height
      });

      detections.forEach(detection => {
        const isSelected = detection.id === selectedDetectionId;
        const { x1, y1, x2, y2 } = detection.bbox;

        console.log('Detection bbox:', { x1, y1, x2, y2, class: detection.class });

        // Coordinates are normalized (0-1), multiply by displayed dimensions
        const scaledX1 = x1 * displayedWidth;
        const scaledY1 = y1 * displayedHeight;
        const scaledX2 = x2 * displayedWidth;
        const scaledY2 = y2 * displayedHeight;
        const scaledWidth = scaledX2 - scaledX1;
        const scaledHeight = scaledY2 - scaledY1;

        console.log('Scaled bbox:', { scaledX1, scaledY1, scaledWidth, scaledHeight });

        ctx.strokeStyle = isSelected ? '#3b82f6' : '#10b981';
        ctx.lineWidth = isSelected ? 4 : 3;
        ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);

        ctx.fillStyle = isSelected ? '#3b82f6' : '#10b981';
        ctx.globalAlpha = 0.2;
        ctx.fillRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
        ctx.globalAlpha = 1.0;

        const label = `${detection.class} ${Math.round(detection.confidence * 100)}%`;
        ctx.font = '14px Inter, system-ui, sans-serif';
        const textMetrics = ctx.measureText(label);
        const textHeight = 20;

        ctx.fillStyle = isSelected ? '#3b82f6' : '#10b981';
        ctx.fillRect(
          scaledX1,
          scaledY1 - textHeight - 4,
          textMetrics.width + 12,
          textHeight + 4
        );

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, scaledX1 + 6, scaledY1 - 8);
      });
    };

    image.onload = drawDetections;
    drawDetections();

    const resizeObserver = new ResizeObserver(drawDetections);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
    };
  }, [imageUrl, detections, selectedDetectionId]);

  return (
    <div ref={containerRef} className="relative w-full">
      <img
        ref={imageRef}
        src={imageUrl}
        alt="Detection preview"
        className="w-full h-auto max-h-80 object-contain rounded-lg shadow-md"
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 pointer-events-none"
      />
    </div>
  );
}
