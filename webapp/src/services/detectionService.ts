import { Detection } from '../types/detection';

const API_BASE_URL = 'http://localhost:8000';

export interface DetectionResponse {
  detections: Detection[];
  annotatedImage: string;
}

export async function detectObjects(imageFile: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Detection failed: ${response.statusText}`);
  }

  const data = await response.json();

  // Transform API response to match our Detection interface
  const detections: Detection[] = data.detections.map((det: any, index: number) => ({
    id: `${index}`,
    class: det.class,
    class_id: det.class_id,
    confidence: det.confidence,
    cropped_image: det.cropped_image,
  }));

  return {
    detections,
    annotatedImage: data.annotated_image,
  };
}
