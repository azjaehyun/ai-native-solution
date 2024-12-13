// logic.js
import { useState, useRef, useEffect, useCallback } from "react";
import { getChatHistory, saveChatHistory } from '../indexdb/indexedDB';

export const useChatLogic = () => {
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState("ChatGPT 4.0");
  const [currentChat, setCurrentChat] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isWaitingForServer, setIsWaitingForServer] = useState(false);
  const [isSnackbarOpen, setSnackbarOpen] = useState(false);
  const [isAlertOpen, setAlertOpen] = useState(false);
  const [isChatHistoryEnabled, setIsChatHistoryEnabled] = useState(true);
  const [isEditing, setIsEditing] = useState(null);
  const [newTitle, setNewTitle] = useState("");
  const [selectedChatIndex, setSelectedChatIndex] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  
  const timeoutRef = useRef(null);
  const controllerRef = useRef(null);
  const chatBoxRef = useRef(null);
  const ws = useRef(null);

  const chatHistoryLimit = 2;

  const connectWebSocket = useCallback(() => {
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return;
    }
  
    ws.current = new WebSocket('ws://localhost:8765');
  
    const connectionTimeout = setTimeout(() => {
      if (ws.current.readyState !== WebSocket.OPEN) {
        ws.current.close();
      }
    }, 3000);
  
    ws.current.onopen = () => {
      clearTimeout(connectionTimeout);
      console.log('WebSocket connection established');
      setSnackbarOpen(true);
      setCurrentChat([{ sender: "server", text: "무엇을 도와드릴까요?" }]);
      setTimeout(() => {
        setSnackbarOpen(false);
      }, 3000);
    };
  
    ws.current.onmessage = (event) => {
      const serverMessage = event.data;
      console.log("Received server message:", serverMessage);
  
      setCurrentChat((prevChat) => [...prevChat, { sender: "server", text: serverMessage }]);
      setIsWaitingForServer(false);
      updateChatHistory(serverMessage, "server");
    };
  
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  
    ws.current.onclose = () => {
      console.log('WebSocket connection closed, retrying in 20 seconds...');
      setTimeout(connectWebSocket, 20000);
    };
  }, []);

  useEffect(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ model: selectedModel }));
    }
  }, [selectedModel]);

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connectWebSocket]);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [currentChat]);

  useEffect(() => {
    const fetchChatHistory = async () => {
      const history = await getChatHistory();
      setChatHistory(history);
    };
    fetchChatHistory();
  }, []);

  const toggleSidebar = () => {
    setSidebarOpen(!isSidebarOpen);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleSend = () => {
    if (!inputText.trim() || isWaitingForServer) return;
  
    const newUserMessage = inputText;
    setCurrentChat((prevChat) => [...prevChat, { sender: "user", text: newUserMessage }]);
    setInputText("");
    setIsWaitingForServer(true);
  
    const sendMessage = () => {
      if (ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(newUserMessage);
        console.log("Sent message to server:", newUserMessage);
      } else {
        console.error("WebSocket is not open, retrying connection...");
        connectWebSocket();
      }
    };
  
    if (ws.current.readyState !== WebSocket.OPEN) {
      connectWebSocket();
      setTimeout(sendMessage, 3000);
    } else {
      sendMessage();
    }
  
    setChatHistory((prevHistory) => {
      if (selectedChatIndex === null) {
        const newChatIndex = prevHistory.length;
        const newChatTitle = `새로운 채팅 ${new Date().toLocaleString()}`;
        const newChat = {
          index: newChatIndex,
          date: new Date().toLocaleDateString(),
          titles: [newChatTitle],
          messages: [{ sender: "user", text: newUserMessage }],
        };
  
        setSelectedChatIndex(newChatIndex);
        saveChatHistory([...prevHistory, newChat]);
        return [...prevHistory, newChat];
      } else {
        const updatedHistory = prevHistory.map((chat) => {
          if (chat.index === selectedChatIndex) {
            return { ...chat, messages: [...chat.messages, { sender: "user", text: newUserMessage }] };
          }
          return chat;
        });
  
        saveChatHistory(updatedHistory);
        return updatedHistory;
      }
    });
  };

  const updateChatHistory = (newMessage, sender) => {
    setChatHistory((prevHistory) => {
      let currentChatIndex = selectedChatIndex;
  
      if (currentChatIndex === null) {
        currentChatIndex = prevHistory.length > 0 ? prevHistory.length - 1 : 0;
        setSelectedChatIndex(currentChatIndex);
      }
  
      const chatExists = prevHistory.some(chat => chat.index === currentChatIndex);
  
      if (chatExists) {
        const updatedHistory = prevHistory.map((chat) => {
          if (chat.index === currentChatIndex) {
            return { ...chat, messages: [...chat.messages, { sender, text: newMessage }] };
          }
          return chat;
        });
  
        saveChatHistory(updatedHistory);
        return updatedHistory;
      } else {
        const newChatTitle = new Date().toLocaleString();
        const newChat = {
          index: currentChatIndex,
          date: new Date().toLocaleDateString(),
          titles: [newChatTitle],
          messages: [{ sender, text: newMessage }],
        };
  
        const updatedHistory = [...prevHistory, newChat];
        saveChatHistory(updatedHistory);
        return updatedHistory;
      }
    });
  };

  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSend();
    }
  };

  const handleStopRequest = () => {
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    setIsWaitingForServer(false);
    clearTimeout(timeoutRef.current);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      alert(`파일 업로드: ${file.name}`);
    }
  };

  const handleChatSelect = (index) => {
    setSelectedChatIndex(index);
  
    setChatHistory((prevHistory) => {
      const selectedChat = prevHistory.find(chat => chat.index === index);
      if (selectedChat) {
        setCurrentChat(selectedChat.messages || []);
      }
      return prevHistory;
    });
  };

  const handleNewChat = () => {
    if (chatHistory.length >= chatHistoryLimit) {
      setAlertOpen(true);
      return;
    }
    const newChatTitle = `새로운 채팅 ${new Date().toLocaleString()}`;
    const newChat = { 
      index: chatHistory.length,
      date: new Date().toLocaleDateString(), 
      titles: [newChatTitle], 
      messages: [] 
    };
    setChatHistory((prevHistory) => {
      const newHistory = [...prevHistory, newChat];
      saveChatHistory(newHistory);
      return newHistory;
    });
    setCurrentChat([
      { sender: "server", text: "새로운 채팅이 시작되었습니다." },
      { sender: "server", text: "무엇을 도와드릴까요?" }
    ]);
    setSelectedChatIndex(newChat.index);
  };

  const toggleChatHistory = () => {
    setIsChatHistoryEnabled(!isChatHistoryEnabled);
  };

  const handleEditTitle = (index, event) => {
    event.stopPropagation();
    setIsEditing(index);
    setNewTitle(chatHistory[index].titles[0]);
  };

  const handleSaveTitle = (index) => {
    const updatedHistory = chatHistory.map((chat, i) => {
      if (i === index) {
        return { ...chat, titles: [newTitle] };
      }
      return chat;
    });
    setChatHistory(updatedHistory);
    saveChatHistory(updatedHistory);
    setIsEditing(null);
  };

  const groupedChatHistory = chatHistory.reduce((acc, chat) => {
    const date = chat.date;
    if (!acc[date]) {
      acc[date] = [];
    }
    acc[date].push(chat.titles[0]);
    return acc;
  }, {});

  return {
    isSidebarOpen,
    toggleSidebar,
    selectedModel,
    handleModelChange,
    currentChat,
    inputText,
    setInputText,
    handleKeyPress,
    handleSend,
    handleStopRequest,
    isWaitingForServer,
    handleFileUpload,
    isSnackbarOpen,
    setSnackbarOpen,
    isAlertOpen,
    setAlertOpen,
    isChatHistoryEnabled,
    toggleChatHistory,
    isEditing,
    setIsEditing,
    newTitle,
    setNewTitle,
    selectedChatIndex,
    setSelectedChatIndex,
    chatHistory,
    setChatHistory,
    handleChatSelect,
    handleNewChat,
    handleEditTitle,
    handleSaveTitle,
    groupedChatHistory,
    chatBoxRef
  };
};