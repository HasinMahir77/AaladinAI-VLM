export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  id: string;
  label: string;
  confidence: number;
  bbox: BoundingBox;
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
