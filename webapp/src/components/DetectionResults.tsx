import { useEffect, useRef } from 'react';
import { Check } from 'lucide-react';
import { Detection } from '../types/detection';

interface DetectionResultsProps {
  imageUrl: string;
  detections: Detection[];
  selectedDetectionId?: string;
  onSelectDetection: (id: string) => void;
}

export default function DetectionResults({
  imageUrl,
  detections,
  selectedDetectionId,
  onSelectDetection
}: DetectionResultsProps) {
  const cropImage = (detection: Detection): string => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = imageUrl;

    const { x, y, width, height } = detection.bbox;
    canvas.width = width;
    canvas.height = height;

    if (ctx) {
      ctx.drawImage(img, x, y, width, height, 0, 0, width, height);
    }

    return canvas.toDataURL();
  };

  return (
    <div className="mt-6">
      <h3 className="text-lg font-bold text-gray-800 mb-3">
        Select a Detection ({detections.length} found)
      </h3>

      <div className="grid grid-cols-2 gap-3">
        {detections.map((detection) => {
          const isSelected = detection.id === selectedDetectionId;

          return (
            <button
              key={detection.id}
              onClick={() => onSelectDetection(detection.id)}
              className={`
                relative p-2 rounded-lg border-2 transition-all duration-200
                hover:scale-105 hover:shadow-md
                ${isSelected
                  ? 'border-blue-500 bg-blue-50 shadow-md'
                  : 'border-gray-200 bg-white hover:border-blue-300'
                }
              `}
            >
              <div className="aspect-square bg-gray-100 rounded overflow-hidden mb-1.5">
                <DetectionThumbnail
                  imageUrl={imageUrl}
                  detection={detection}
                />
              </div>

              <div className="text-left">
                <p className="font-semibold text-xs text-gray-800 truncate">
                  {detection.label}
                </p>
                <p className="text-xs text-gray-500">
                  {Math.round(detection.confidence * 100)}%
                </p>
              </div>

              {isSelected && (
                <div className="absolute top-1.5 right-1.5 bg-blue-500 text-white rounded-full p-0.5">
                  <Check size={14} />
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function DetectionThumbnail({ imageUrl, detection }: { imageUrl: string; detection: Detection }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      const { x, y, width, height } = detection.bbox;

      canvas.width = 200;
      canvas.height = 200;

      const scale = Math.min(200 / width, 200 / height);
      const scaledWidth = width * scale;
      const scaledHeight = height * scale;
      const offsetX = (200 - scaledWidth) / 2;
      const offsetY = (200 - scaledHeight) / 2;

      ctx.fillStyle = '#f3f4f6';
      ctx.fillRect(0, 0, 200, 200);

      ctx.drawImage(
        img,
        x, y, width, height,
        offsetX, offsetY, scaledWidth, scaledHeight
      );
    };

    img.src = imageUrl;
  }, [imageUrl, detection]);

  return <canvas ref={canvasRef} className="w-full h-full object-cover" />;
}
