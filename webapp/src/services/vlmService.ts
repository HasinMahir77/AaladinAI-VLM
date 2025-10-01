import { Detection } from '../types/detection';

const responses = [
  "Based on the detection results, this object appears to be clearly visible in the image with high confidence. The bounding box coordinates indicate its precise location.",
  "That's an interesting observation! The detected object shows distinctive features that are commonly associated with this category.",
  "From a visual analysis perspective, the object exhibits typical characteristics. The detection confidence suggests a strong match with the classification.",
  "The spatial positioning of this object in the image is quite noteworthy. It's located in a region that provides good context for understanding the overall scene.",
  "Looking at the detection data, the object boundaries are well-defined, which helps in accurate identification. The confidence score reflects the clarity of the detection.",
  "This is a great question about the detected object! The features captured within the bounding box align well with the expected visual patterns.",
  "The object's placement and the surrounding context in the image provide valuable information. Detection algorithms typically perform well with such clear instances.",
  "Interesting point! The detected region contains distinctive visual elements that contribute to the classification. The confidence level suggests reliable detection.",
  "Based on the bounding box coordinates and the detected features, this object stands out clearly in the scene. The detection system has identified key visual markers.",
  "That's a thoughtful question! The detected object's characteristics align with typical examples in this category, which explains the high confidence score."
];

export async function chatWithVLM(
  userMessage: string,
  selectedDetection: Detection,
  imageUrl: string
): Promise<string> {
  await new Promise(resolve => setTimeout(resolve, 1500));

  const detectionContext = `Detection: ${selectedDetection.label} (${Math.round(selectedDetection.confidence * 100)}% confidence)`;

  const responseIndex = Math.floor(Math.random() * responses.length);
  const baseResponse = responses[responseIndex];

  if (userMessage.toLowerCase().includes('what') || userMessage.toLowerCase().includes('describe')) {
    return `I can see a ${selectedDetection.label} in the image with ${Math.round(selectedDetection.confidence * 100)}% confidence. ${baseResponse}`;
  }

  if (userMessage.toLowerCase().includes('where') || userMessage.toLowerCase().includes('location')) {
    return `The ${selectedDetection.label} is located at coordinates (${Math.round(selectedDetection.bbox.x)}, ${Math.round(selectedDetection.bbox.y)}) with dimensions ${Math.round(selectedDetection.bbox.width)}x${Math.round(selectedDetection.bbox.height)} pixels. ${baseResponse}`;
  }

  if (userMessage.toLowerCase().includes('confidence') || userMessage.toLowerCase().includes('sure')) {
    return `The detection confidence for this ${selectedDetection.label} is ${Math.round(selectedDetection.confidence * 100)}%, which indicates a ${selectedDetection.confidence > 0.9 ? 'very high' : 'good'} level of certainty. ${baseResponse}`;
  }

  return `Regarding the ${selectedDetection.label}: ${baseResponse}`;
}
