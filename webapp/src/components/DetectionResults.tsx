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
              <div className="bg-gray-100 rounded overflow-hidden mb-1.5 flex items-center justify-center min-h-[120px]">
                <DetectionThumbnail
                  imageUrl={imageUrl}
                  detection={detection}
                />
              </div>

              <div className="text-left">
                <p className="font-semibold text-xs text-gray-800 truncate">
                  {detection.class}
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
  // Use the base64 cropped image from the API
  const croppedImageSrc = `data:image/jpeg;base64,${detection.cropped_image}`;

  return (
    <img
      src={croppedImageSrc}
      alt={detection.class}
      className="max-w-full max-h-[200px] object-contain"
    />
  );
}
