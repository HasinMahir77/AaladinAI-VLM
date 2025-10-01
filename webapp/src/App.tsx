import { useState } from 'react';
import { Eye, X } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import DetectionResults from './components/DetectionResults';
import ChatInterface from './components/ChatInterface';
import { Detection, ChatMessage } from './types/detection';
import { detectObjects } from './services/detectionService';
import { chatWithVLM } from './services/vlmService';

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState<string>('');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string>('');
  const [isDetecting, setIsDetecting] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);

  const handleImageSelected = (file: File, url: string) => {
    setImageFile(file);
    setImageUrl(url);
    setAnnotatedImageUrl('');
    setDetections([]);
    setSelectedDetectionId('');
    setShowChat(false);
    setMessages([]);
  };

  const handleRemoveImage = () => {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    setImageFile(null);
    setImageUrl('');
    setAnnotatedImageUrl('');
    setDetections([]);
    setSelectedDetectionId('');
    setShowChat(false);
    setMessages([]);
  };

  const handleDetect = async () => {
    if (!imageFile) return;

    setIsDetecting(true);
    setDetections([]);
    setAnnotatedImageUrl('');
    setSelectedDetectionId('');
    setShowChat(false);
    setMessages([]);

    try {
      const results = await detectObjects(imageFile);

      if (results.detections.length === 0) {
        alert('No objects detected in the image. Please try another image.');
        setIsDetecting(false);
        return;
      }

      setDetections(results.detections);
      setAnnotatedImageUrl(`data:image/jpeg;base64,${results.annotatedImage}`);
      setShowChat(true);
    } catch (error) {
      console.error('Detection failed:', error);
      alert('Detection failed. Please try again.');
    } finally {
      setIsDetecting(false);
    }
  };

  const handleSelectDetection = async (id: string) => {
    setSelectedDetectionId(id);
    setMessages([]);

    const selectedDetection = detections.find(d => d.id === id);
    if (!selectedDetection) return;

    setIsLoadingResponse(true);

    try {
      const description = await chatWithVLM('', selectedDetection, imageUrl);

      const assistantMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: description,
        timestamp: Date.now()
      };

      setMessages([assistantMessage]);
    } catch (error) {
      console.error('Description failed:', error);
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error generating the description. Please try again.',
        timestamp: Date.now()
      };
      setMessages([errorMessage]);
    } finally {
      setIsLoadingResponse(false);
    }
  };

  const isChatEnabled = detections.length > 0 && selectedDetectionId;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <header className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg shadow-lg">
              <Eye size={24} className="text-white" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
              YOLO v12 with VLM
            </h1>
          </div>
          <p className="text-gray-600">
            Upload an image, detect people/vehicles, and get a detailed description of the image.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Image Upload</h2>
              {annotatedImageUrl ? (
                <div className="relative">
                  <img
                    src={annotatedImageUrl}
                    alt="Detection preview with bounding boxes"
                    className="w-full h-auto max-h-80 object-contain rounded-lg shadow-md"
                  />
                  <button
                    onClick={handleRemoveImage}
                    className="absolute top-2 right-2 bg-white hover:bg-red-50 text-red-600 p-1.5 rounded-full shadow-lg transition-all duration-200 hover:scale-110"
                    title="Change image"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <ImageUpload
                  onImageSelected={handleImageSelected}
                  currentImage={imageUrl}
                  onRemoveImage={handleRemoveImage}
                />
              )}

              {imageUrl && !detections.length && (
                <div className="mt-4 text-center">
                  <button
                    onClick={handleDetect}
                    disabled={isDetecting}
                    className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-cyan-600 disabled:from-gray-300 disabled:to-gray-400 transition-all duration-200 hover:shadow-lg hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed"
                  >
                    {isDetecting ? (
                      <span className="flex items-center gap-2">
                        <div className="w-4 h-4 border-3 border-white border-t-transparent rounded-full animate-spin" />
                        Detecting Objects...
                      </span>
                    ) : (
                      <span className="flex items-center gap-2">
                        <Eye size={20} />
                        Detect
                      </span>
                    )}
                  </button>
                </div>
              )}
            </div>

            {detections.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Detection Results</h2>
                <DetectionResults
                  imageUrl={imageUrl}
                  detections={detections}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={handleSelectDetection}
                />
              </div>
            )}
          </div>

          <div className="lg:sticky lg:top-6 lg:self-start">
            <ChatInterface
              onSendMessage={async () => {}}
              messages={messages}
              isLoading={isLoadingResponse}
              disabled={!isChatEnabled}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
