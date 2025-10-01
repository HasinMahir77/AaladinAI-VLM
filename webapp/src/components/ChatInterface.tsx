import { useRef, useEffect } from 'react';
import { Bot, User, Loader2 } from 'lucide-react';
import { ChatMessage } from '../types/detection';

interface ChatInterfaceProps {
  onSendMessage: (message: string) => Promise<void>;
  messages: ChatMessage[];
  isLoading: boolean;
  disabled?: boolean;
}

export default function ChatInterface({ onSendMessage, messages, isLoading, disabled = false }: ChatInterfaceProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className={`bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200 ${disabled ? 'opacity-50' : ''}`}>
      <div className={`bg-gradient-to-r from-blue-500 to-cyan-500 text-white p-4 ${disabled ? 'grayscale' : ''}`}>
        <div className="flex items-center gap-3">
          <Bot size={20} />
          <div>
            <h2 className="text-lg font-bold">VLM Description</h2>
            <p className="text-xs text-blue-50">AI-generated description of detected object</p>
          </div>
        </div>
      </div>

      <div className="h-96 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <Bot size={40} className="mx-auto mb-2 opacity-50" />
              <p className="text-sm">
                {disabled
                  ? 'Upload an image, run detection, and select an object'
                  : 'Select a detected object to see its description'
                }
              </p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-2 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {message.role === 'assistant' && (
              <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                <Bot size={16} className="text-white" />
              </div>
            )}

            <div
              className={`max-w-[75%] rounded-lg px-3 py-2 text-sm ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
              }`}
            >
              <p className="whitespace-pre-wrap break-words">{message.content}</p>
            </div>

            {message.role === 'user' && (
              <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gray-600 flex items-center justify-center">
                <User size={16} className="text-white" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-2 justify-start">
            <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Bot size={16} className="text-white" />
            </div>
            <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 shadow-sm">
              <div className="flex items-center gap-2">
                <Loader2 size={14} className="animate-spin text-blue-500" />
                <span className="text-gray-500 text-sm">Thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

    </div>
  );
}
