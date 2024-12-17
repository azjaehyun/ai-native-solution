import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  AppBar,
  Toolbar,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Box,
  Typography,
  Select,
  MenuItem,
  TextField,
  CircularProgress,
  Avatar,
  Snackbar,
  Alert,
  Tooltip,
  Switch,
  FormControlLabel,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import SendIcon from "@mui/icons-material/Send";
import NoteAddIcon from "@mui/icons-material/LocalHospital";
import SmartToyTwoToneIcon from "@mui/icons-material/SmartToyTwoTone";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import { getChatHistory, saveChatHistory } from './indexdb/indexedDB';
import axios from 'axios';

const App = () => {
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

  const chatHistoryLimit = 2;

  useEffect(() => {
    const loadChatHistory = async () => {
      const history = await getChatHistory();
      setChatHistory(history);
    };
    loadChatHistory();
  }, []);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [currentChat]);

  const sendMessageToServer = async (message) => {
    try {
      const response = await axios.post('http://localhost:8765/echo', { message });
      return response.data.resultData.message;
    } catch (error) {
      console.error('Error sending message to server:', error);
      return 'Error: Unable to send message to server';
    }
  };

  const handleSend = async () => {
    if (!inputText.trim() || isWaitingForServer) return;

    if (selectedChatIndex === null && chatHistory.length >= chatHistoryLimit) {
      setAlertOpen(true);
      return;
    }

    const newUserMessage = inputText;
    setCurrentChat((prevChat) => [...prevChat, { sender: "user", text: newUserMessage }]);
    setInputText("");
    setIsWaitingForServer(true);

    const serverMessage = await sendMessageToServer(newUserMessage);
    setCurrentChat((prevChat) => [...prevChat, { sender: "server", text: serverMessage }]);
    setIsWaitingForServer(false);

    setChatHistory((prevHistory) => {
      if (selectedChatIndex === null) {
        const newChatIndex = prevHistory.length;
        const newChatTitle = `새로운 채팅 ${new Date().toLocaleString()}`;
        const newChat = {
          index: newChatIndex,
          date: new Date().toLocaleDateString(),
          titles: [newChatTitle],
          messages: [
            { sender: "user", text: newUserMessage },
            { sender: "server", text: serverMessage }
          ],
        };

        setSelectedChatIndex(newChatIndex);
        saveChatHistory([...prevHistory, newChat]);
        return [...prevHistory, newChat];
      } else {
        const updatedHistory = prevHistory.map((chat) => {
          if (chat.index === selectedChatIndex) {
            return {
              ...chat,
              messages: [
                ...chat.messages,
                { sender: "user", text: newUserMessage },
                { sender: "server", text: serverMessage }
              ],
            };
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

  const toggleSidebar = () => {
    setSidebarOpen(!isSidebarOpen);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleStopRequest = () => {
    if (controllerRef.current) {
      controllerRef.current.abort();
      setIsWaitingForServer(false);
    }
  };

  const groupedChatHistory = chatHistory.reduce((acc, chat) => {
    const date = chat.date;
    if (!acc[date]) {
      acc[date] = [];
    }
    acc[date].push(chat.titles[0]);
    return acc;
  }, {});

  return (
    <Box sx={{ position: "relative", width: "100vw", height: "100vh", bgcolor: "#1e1e1e" }}>
      {isSidebarOpen && (
        <Drawer
          variant="persistent"
          open={isSidebarOpen}
          sx={{
            "& .MuiDrawer-paper": {
              position: "absolute",
              width: "250px",
              height: "100%",
              boxSizing: "border-box",
              bgcolor: "#333",
              color: "#fff",
              margin: 0,
            },
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", p: 2 }}>
            <img
              src="/logo.png"
              alt="Company Logo"
              style={{ width: "40px", height: "40px", marginRight: "10px" }}
            />
            <Typography variant="h6" sx={{ color: "#fff", fontWeight: "bold", flexGrow: 1 }}>
              BespinGlobal
            </Typography>
            <Tooltip title="새로운 채팅">
              <IconButton color="inherit" onClick={handleNewChat}>
                <NoteAddIcon />
              </IconButton>
            </Tooltip>
          </Box>

          <Box sx={{ display: "flex", justifyContent: "flex-end", p: 1 }}>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleSidebar}
            >
              <MenuIcon />
            </IconButton>
          </Box>

          <Box sx={{ display: "flex", justifyContent: "center", p: 1 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={isChatHistoryEnabled}
                  onChange={toggleChatHistory}
                  color="primary"
                />
              }
              label="채팅 히스토리"
              sx={{ color: "#fff" }}
            />
          </Box>

          {isChatHistoryEnabled && (
            <List>
              {Object.keys(groupedChatHistory).map((date, dateIndex) => (
                <Box key={dateIndex} sx={{ mb: 2 }}>
                  <Typography sx={{ px: 2, fontWeight: "bold", color: "#aaa" }}>
                    {date}
                  </Typography>
                  {groupedChatHistory[date].map((title, titleIndex) => (
                    <ListItem key={titleIndex} button="true" onClick={() => handleChatSelect(titleIndex)}>
                      <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        {isEditing === titleIndex ? (
                          <TextField
                            value={newTitle}
                            onChange={(e) => setNewTitle(e.target.value)}
                            onBlur={() => handleSaveTitle(titleIndex)}
                            onKeyPress={(e) => {
                              if (e.key === "Enter") {
                                handleSaveTitle(titleIndex);
                              }
                            }}
                            onClick={(event) => event.stopPropagation()}
                            autoFocus
                            sx={{ bgcolor: "#f0f0f0", borderRadius: 3, flexGrow: 1 }}
                          />
                        ) : (
                          <>
                            <ListItemText primary={title} sx={{ color: titleIndex === selectedChatIndex ? "#a8dadc" : "#fff", flexGrow: 1 }} />
                            <IconButton onClick={(event) => { event.stopPropagation(); handleEditTitle(titleIndex, event); }}>
                              <MoreVertIcon />
                            </IconButton>
                          </>
                        )}
                      </div>
                    </ListItem>
                  ))}
                </Box>
              ))}
            </List>
          )}
        </Drawer>
      )}

      <Box
        sx={{
          position: "absolute",
          top: 0,
          left: isSidebarOpen ? "250px" : "0",
          width: isSidebarOpen ? "calc(100% - 250px)" : "100%",
          height: "100%",
          transition: "left 0.3s ease, width 0.3s ease",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <AppBar position="static" sx={{ bgcolor: "#333" }}>
          <Toolbar>
            {!isSidebarOpen && (
              <IconButton
                edge="start"
                color="inherit"
                aria-label="menu"
                onClick={toggleSidebar}
                sx={{ mr: 2 }}
              >
                <MenuIcon />
              </IconButton>
            )}
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              AWS Bedrock POC - AI Native Team
            </Typography>
            <Select
              value={selectedModel}
              onChange={handleModelChange}
              sx={{ bgcolor: "#444", color: "#fff", borderRadius: 1 }}
            >
              <MenuItem value="ChatGPT 4.0">ChatGPT 4.0</MenuItem>
              <MenuItem value="ChatGPT 3.5">ChatGPT 3.5</MenuItem>
            </Select>
          </Toolbar>
        </AppBar>

        <Box
          ref={chatBoxRef}
          sx={{
            flexGrow: 1,
            display: "flex",
            flexDirection: "column",
            p: 2,
            bgcolor: "#1e1e1e",
            overflowY: "auto",
            position: "relative",
            "&::-webkit-scrollbar": {
              width: "8px",
            },
            "&::-webkit-scrollbar-thumb": {
              backgroundColor: "#555",
              borderRadius: "4px",
            },
            "&::-webkit-scrollbar-track": {
              backgroundColor: "#111",
            },
          }}
        >
          {isWaitingForServer && (
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                zIndex: 10,
              }}
            >
              <CircularProgress color="info" />
            </Box>
          )}

          {currentChat.map((msg, idx) =>
            msg.sender === "server" ? (
              <Box
                key={idx}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  margin: "5px 0",
                  textAlign: "left",
                }}
              >
                <Avatar sx={{ bgcolor: "#1976d2", mr: 1 }}>
                  <SmartToyTwoToneIcon />
                </Avatar>
                <Typography
                  sx={{
                    bgcolor: "#444",
                    color: "#fff",
                    p: 1,
                    borderRadius: 2,
                    maxWidth: "70%",
                  }}
                >
                  {msg.text}
                </Typography>
              </Box>
            ) : (
              <Typography
                key={idx}
                sx={{
                  margin: "5px 0",
                  alignSelf: "flex-end",
                  bgcolor: "#1976d2",
                  color: "#fff",
                  p: 1,
                  borderRadius: 2,
                  maxWidth: "70%",
                  textAlign: "center",
                }}
              >
                {msg.text}
              </Typography>
            )
          )}
        </Box>

        <Box sx={{ display: "flex", alignItems: "center", p: 1 }}>
          <TextField
            variant="outlined"
            fullWidth
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            sx={{ bgcolor: "#fff", borderRadius: 1 }}
            placeholder="메시지를 입력하세요"
            InputProps={{
              endAdornment: (
                <>
                  <label htmlFor="file-upload">
                    <IconButton component="span" sx={{ ml: 1 }}>
                      <UploadFileIcon />
                    </IconButton>
                  </label>
                  <input
                    id="file-upload"
                    type="file"
                    style={{ display: "none" }}
                    onChange={handleFileUpload}
                  />
                  <IconButton onClick={handleSend} sx={{ ml: 1 }}>
                    <SendIcon />
                  </IconButton>
                </>
              ),
            }}
          />
          {isWaitingForServer && (
            <IconButton onClick={handleStopRequest} sx={{ ml: 1 }}>
              <img
                src="/stop-icon.png"
                alt="Stop Button"
                style={{ width: "50px", height: "50px" }}
              />
            </IconButton>
          )}
        </Box>
      </Box>
      <Snackbar
        open={isSnackbarOpen}
        autoHideDuration={3000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setSnackbarOpen(false)} severity="success" sx={{ width: '100%' }}>
          WebSocket connection established
        </Alert>
      </Snackbar>
      <Snackbar
        open={isAlertOpen}
        autoHideDuration={3000}
        onClose={() => setAlertOpen(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setAlertOpen(false)} severity="warning" sx={{ width: '100%' }}>
          채팅창을 최대 {chatHistoryLimit}개 까지만 생성 할 수 있습니다.
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default App;