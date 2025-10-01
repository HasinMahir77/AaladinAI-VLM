export interface Detection {
  id: string;
  class: string;
  class_id: number;
  confidence: number;
  cropped_image: string;
}

export interface DetectionResult {
  imageUrl: string;
  detections: Detection[];
  timestamp: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}
