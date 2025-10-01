import { Detection } from '../types/detection';

const API_BASE_URL = 'http://localhost:8000';

export async function startChatSession(
  selectedDetection: Detection
): Promise<{ sessionId: string; description: string }> {
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

  // Send to /start-chat endpoint
  const formData = new FormData();
  formData.append('file', croppedFile);

  const response = await fetch(`${API_BASE_URL}/start-chat`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to start chat session: ${response.statusText}`);
  }

  const data = await response.json();

  return {
    sessionId: data.session_id,
    description: data.description,
  };
}

export async function sendChatMessage(
  sessionId: string,
  message: string
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat message failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.response;
}
