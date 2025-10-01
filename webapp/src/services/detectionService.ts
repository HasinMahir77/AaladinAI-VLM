import { Detection } from '../types/detection';

export async function detectObjects(imageFile: File): Promise<Detection[]> {
  await new Promise(resolve => setTimeout(resolve, 2000));

  const img = new Image();
  const imageUrl = URL.createObjectURL(imageFile);

  return new Promise((resolve) => {
    img.onload = () => {
      const width = img.naturalWidth;
      const height = img.naturalHeight;

      const mockDetections: Detection[] = [
        {
          id: '1',
          label: 'Person',
          confidence: 0.95,
          bbox: {
            x: width * 0.2,
            y: height * 0.15,
            width: width * 0.25,
            height: height * 0.5
          }
        },
        {
          id: '2',
          label: 'Car',
          confidence: 0.89,
          bbox: {
            x: width * 0.55,
            y: height * 0.4,
            width: width * 0.3,
            height: height * 0.35
          }
        },
        {
          id: '3',
          label: 'Dog',
          confidence: 0.87,
          bbox: {
            x: width * 0.1,
            y: height * 0.6,
            width: width * 0.2,
            height: height * 0.25
          }
        },
        {
          id: '4',
          label: 'Building',
          confidence: 0.92,
          bbox: {
            x: width * 0.7,
            y: height * 0.05,
            width: width * 0.25,
            height: height * 0.4
          }
        }
      ];

      URL.revokeObjectURL(imageUrl);
      resolve(mockDetections);
    };

    img.src = imageUrl;
  });
}
