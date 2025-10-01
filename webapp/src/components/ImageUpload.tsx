import { useState, useRef } from 'react';
import { Upload, X } from 'lucide-react';

interface ImageUploadProps {
  onImageSelected: (file: File, imageUrl: string) => void;
  currentImage?: string;
  onRemoveImage?: () => void;
}

export default function ImageUpload({ onImageSelected, currentImage, onRemoveImage }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
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
  );
}
