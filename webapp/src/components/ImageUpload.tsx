import { useState, useRef } from 'react';
import { Upload, X, Link } from 'lucide-react';

interface ImageUploadProps {
  onImageSelected: (file: File, imageUrl: string) => void;
  currentImage?: string;
  onRemoveImage?: () => void;
}

export default function ImageUpload({ onImageSelected, currentImage, onRemoveImage }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [useUrl, setUseUrl] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [isLoadingUrl, setIsLoadingUrl] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('Image size should be less than 10MB');
      return;
    }

    const imageUrl = URL.createObjectURL(file);
    onImageSelected(file, imageUrl);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleUrlSubmit = async () => {
    if (!urlInput.trim()) return;

    setIsLoadingUrl(true);
    try {
      const response = await fetch(urlInput);
      if (!response.ok) throw new Error('Failed to fetch image');

      const blob = await response.blob();
      if (!blob.type.startsWith('image/')) {
        alert('URL does not point to a valid image');
        return;
      }

      const file = new File([blob], 'url-image.jpg', { type: blob.type });
      const imageUrl = URL.createObjectURL(file);
      onImageSelected(file, imageUrl);
      setUrlInput('');
    } catch (error) {
      alert('Failed to load image from URL. Please check the URL and try again.');
      console.error(error);
    } finally {
      setIsLoadingUrl(false);
    }
  };

  if (currentImage) {
    return (
      <div className="relative">
        <img
          src={currentImage}
          alt="Uploaded preview"
          className="w-full h-auto max-h-80 object-contain rounded-lg shadow-md"
        />
        <button
          onClick={onRemoveImage}
          className="absolute top-2 right-2 bg-white hover:bg-red-50 text-red-600 p-1.5 rounded-full shadow-lg transition-all duration-200 hover:scale-110"
          title="Change image"
        >
          <X size={16} />
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-gray-200 pb-2">
        <button
          onClick={() => setUseUrl(false)}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
            !useUrl
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <Upload size={18} />
            Upload File
          </div>
        </button>
        <button
          onClick={() => setUseUrl(true)}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
            useUrl
              ? 'bg-blue-500 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <Link size={18} />
            Use URL
          </div>
        </button>
      </div>

      {useUrl ? (
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleUrlSubmit()}
              placeholder="Enter image URL..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoadingUrl}
            />
            <button
              onClick={handleUrlSubmit}
              disabled={!urlInput.trim() || isLoadingUrl}
              className="px-6 py-3 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all"
            >
              {isLoadingUrl ? (
                <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                'Load'
              )}
            </button>
          </div>
          <p className="text-xs text-gray-500 text-center">
            Enter a direct link to an image (e.g., https://example.com/image.jpg)
          </p>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleClick}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            transition-all duration-300 ease-in-out
            ${isDragging
              ? 'border-blue-500 bg-blue-50 scale-105'
              : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            className="hidden"
          />

          <div className="flex flex-col items-center gap-3">
            <div className={`
              p-4 rounded-full transition-all duration-300
              ${isDragging ? 'bg-blue-100' : 'bg-white'}
            `}>
              <Upload size={32} className={`
                transition-colors duration-300
                ${isDragging ? 'text-blue-500' : 'text-gray-400'}
              `} />
            </div>

            <div>
              <p className="text-base font-semibold text-gray-700 mb-1">
                {isDragging ? 'Drop your image here' : 'Upload an image'}
              </p>
              <p className="text-xs text-gray-500">
                Drag and drop or click to browse
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Supports: JPG, PNG, GIF (max 10MB)
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
