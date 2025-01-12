import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import VoiceRecorder from "./VoiceRecorder";
import { ImageIcon } from 'lucide-react';

interface Message {
  text: string;
  isUser: boolean;
  image?: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSendMessage = async () => {
    const newMessage: Message = { text: inputText, isUser: true };
    let imageBase64: string | null = null;

    if (selectedImage) {
      const reader = new FileReader();
      reader.readAsDataURL(selectedImage);
      reader.onloadend = async () => {
        imageBase64 = reader.result as string;
        newMessage.image = imageBase64;

        setMessages([...messages, newMessage]);

        try {
          const response = await fetch("http://127.0.0.1:5000/LearnBot", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: inputText, image: imageBase64 }),
          });

          const data = await response.json();
          const botResponse = data.response || "Sorry, I didn't understand that.";

          setMessages((prev) => [...prev, { text: botResponse, isUser: false }]);
        } catch (error) {
          console.error("Error communicating with the server:", error);
          setMessages((prev) => [
            ...prev,
            { text: "An error occurred. Please try again later.", isUser: false },
          ]);
        }

        setInputText("");
        setSelectedImage(null);
      };
    } else if (inputText.trim()) {
      newMessage.text = inputText;
      setMessages([...messages, newMessage]);

      try {
        const response = await fetch("http://127.0.0.1:5000/LearnBot", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: inputText }),
        });

        const data = await response.json();
        const botResponse = data.response || "Sorry, I didn't understand that.";

        setMessages((prev) => [...prev, { text: botResponse, isUser: false }]);
      } catch (error) {
        console.error("Error communicating with the server:", error);
        setMessages((prev) => [
          ...prev,
          { text: "An error occurred. Please try again later.", isUser: false },
        ]);
      }

      setInputText("");
    } else if (selectedImage) {
      const reader = new FileReader();
      reader.readAsDataURL(selectedImage);
      reader.onloadend = async () => {
        imageBase64 = reader.result as string;
        newMessage.image = imageBase64;

        setMessages([...messages, newMessage]);

        try {
          const response = await fetch("http://127.0.0.1:5000/LearnBot", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ image: imageBase64 }),
          });

          const data = await response.json();
          const botResponse = data.response || "Sorry, I didn't understand that.";

          setMessages((prev) => [...prev, { text: botResponse, isUser: false }]);
        } catch (error) {
          console.error("Error communicating with the server:", error);
          setMessages((prev) => [
            ...prev,
            { text: "An error occurred. Please try again later.", isUser: false },
          ]);
        }

        setInputText("");
        setSelectedImage(null);
      };
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      // Reset the file input field after selecting an image to allow new uploads
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Reset the file input field
      }
    }
  };

  const triggerImageUpload = () => {
    fileInputRef.current?.click();
  };

  const handleTranscript = (transcript: string) => {
    setInputText(transcript);
  };

  return (
    <div className="bg-yellow-100 rounded-3xl p-6 h-[700px] flex flex-col border-4 border-yellow-300 shadow-lg">
      <div className="flex-grow overflow-y-auto mb-4 p-4 bg-white bg-opacity-50 rounded-2xl">
        {messages.map((message, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20, rotate: 0 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={`mb-2 p-3 rounded-2xl ${
              message.isUser ? "bg-blue-200 ml-auto" : "bg-green-200"
            } max-w-[80%] font-comic text-lg shadow-md transform`}
          >
            {message.image && (
              <div className="mb-2">
                <img src={message.image} alt="Uploaded image" className="rounded-lg max-w-full h-auto" />
              </div>
            )}
            <div>{message.text}</div>
          </motion.div>
        ))}
      </div>
      <div className="flex flex-col space-y-4">
        <div className="flex items-center">
          <Input
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your question here..."
            className="flex-grow mr-2 border-2 border-blue-300 font-comic text-lg rounded-full px-4 py-2 bg-white focus:border-pink-300 focus:ring focus:ring-pink-200 focus:ring-opacity-50"
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
          />
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            ref={fileInputRef}
            className="hidden"
          />
          <Button
            onClick={triggerImageUpload}
            className="bg-green-400 hover:bg-green-500 text-white font-comic text-lg rounded-full p-2 mr-2 transform transition-transform duration-200 hover:scale-105"
          >
            <ImageIcon size={24} />
          </Button>
          <Button
            onClick={handleSendMessage}
            className="bg-pink-400 hover:bg-pink-500 text-white font-comic text-lg rounded-full px-6 transform transition-transform duration-200 hover:scale-105"
          >
            Send
          </Button>
        </div>
        {selectedImage && (
          <div className="flex items-center">
            <span className="text-blue-600 font-comic mr-2">Image selected:</span>
            <img src={URL.createObjectURL(selectedImage)} alt="Selected image" className="w-12 h-12 rounded-lg" />
          </div>
        )}
        <motion.div
          whileHover={{ scale: 1.1, rotate: [0, -10, 10, -10, 10, 0] }}
          transition={{ duration: 0.5 }}
        >
          <VoiceRecorder onTranscript={handleTranscript} />
        </motion.div>
      </div>
    </div>
  );
}
