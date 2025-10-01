import { Detection } from '../types/detection';

const API_BASE_URL = 'http://localhost:8000';

export async function chatWithVLM(
  userMessage: string,
  selectedDetection: Detection,
  imageUrl: string
): Promise<string> {
  // Convert base64 cropped image to Blob
  const base64Data = selectedDetection.cropped_image;
  const byteCharacters = atob(base64Data);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: 'image/jpeg' });

  // Create a file from the blob
  const croppedFile = new File([blob], 'cropped.jpg', { type: 'image/jpeg' });

  // Send to /describe endpoint
  const formData = new FormData();
  formData.append('file', croppedFile);

  const response = await fetch(`${API_BASE_URL}/describe`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`VLM description failed: ${response.statusText}`);
  }

  const data = await response.json();

  // Return just the description without class/confidence
  return data.description;
}
