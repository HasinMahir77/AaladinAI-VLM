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

      const rect = image.getBoundingClientRect();
      canvas.width = image.width;
      canvas.height = image.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const scaleX = image.naturalWidth / image.width;
      const scaleY = image.naturalHeight / image.height;

      detections.forEach(detection => {
        const isSelected = detection.id === selectedDetectionId;
        const { x, y, width, height } = detection.bbox;

        const scaledX = x / scaleX;
        const scaledY = y / scaleY;
        const scaledWidth = width / scaleX;
        const scaledHeight = height / scaleY;

        ctx.strokeStyle = isSelected ? '#3b82f6' : '#10b981';
        ctx.lineWidth = isSelected ? 4 : 3;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

        ctx.fillStyle = isSelected ? '#3b82f6' : '#10b981';
        ctx.globalAlpha = 0.2;
        ctx.fillRect(scaledX, scaledY, scaledWidth, scaledHeight);
        ctx.globalAlpha = 1.0;

        const label = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
        ctx.font = '14px Inter, system-ui, sans-serif';
        const textMetrics = ctx.measureText(label);
        const textHeight = 20;

        ctx.fillStyle = isSelected ? '#3b82f6' : '#10b981';
        ctx.fillRect(
          scaledX,
          scaledY - textHeight - 4,
          textMetrics.width + 12,
          textHeight + 4
        );

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, scaledX + 6, scaledY - 8);
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
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
    </div>
  );
}
