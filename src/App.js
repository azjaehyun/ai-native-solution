import React, { useState, useRef, useEffect, useCallback } from "react";
import { RecoilRoot, useRecoilState } from 'recoil';
import { chatHistoryState } from './recoil/atoms';
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
import SmartToyTwoToneIcon from "@mui/icons-material/SmartToyTwoTone"; // AI 로봇 아이콘
import MoreVertIcon from "@mui/icons-material/MoreVert";

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
  const timeoutRef = useRef(null);
  const controllerRef = useRef(null);
  const chatBoxRef = useRef(null);
  const ws = useRef(null);

  const [chatHistory, setChatHistory] = useRecoilState(chatHistoryState);
  const chatHistoryLimit = 2; // 최대 채팅 히스토리 개수

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
      // if (selectedChatIndex === null) {
      //   console.log("selectedChatIndex is not set");
      //   return;
      // }
  
      const serverMessage = event.data;
      console.log("Received server message:", serverMessage);
  
      // 현재 채팅 화면 업데이트
      setCurrentChat((prevChat) => [...prevChat, { sender: "server", text: serverMessage }]);
  
      setIsWaitingForServer(false);
  
      // Recoil 상태 업데이트
      updateChatHistory(serverMessage, "server");
    };
  
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  
    ws.current.onclose = () => {
      console.log('WebSocket connection closed, retrying in 20 seconds...');
      setTimeout(connectWebSocket, 20000); // 20초 후에 재시도
    };
  }, []);


  useEffect(() => {
    // selectedModel 상태가 변경될 때마다 웹 소켓으로 메시지 전송
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ model: selectedModel }));
    }
  }, []);


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

  const toggleSidebar = () => {
    setSidebarOpen(!isSidebarOpen);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleSend = () => {
    if (!inputText.trim() || isWaitingForServer) return;
  
    // 유저 메시지 추가
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
      setTimeout(sendMessage, 3000); // 3초 후에 재시도
    } else {
      sendMessage();
    }
  
    // Recoil 상태 업데이트
    setChatHistory((prevHistory) => {
      // 최초 채팅인지 확인
      if (selectedChatIndex === null) {
        const newChatIndex = prevHistory.length;
        const newChatTitle = `새로운 채팅 ${new Date().toLocaleString()}`;
        const newChat = {
          index: newChatIndex,
          date: new Date().toLocaleDateString(),
          titles: [newChatTitle],
          messages: [{ sender: "user", text: newUserMessage }],
        };
  
        console.log("Added first chat:", newChat);
  
        // 새로운 타이틀 추가 및 선택
        setSelectedChatIndex(newChatIndex);
        return [...prevHistory, newChat];
      } else {
        // 기존 채팅에 메시지 추가
        const updatedHistory = prevHistory.map((chat) => {
          if (chat.index === selectedChatIndex) {
            return { ...chat, messages: [...chat.messages, { sender: "user", text: newUserMessage }] };
          }
          return chat;
        });
  
        console.log("Updated chat history:", updatedHistory);
        return updatedHistory;
      }
    });
  };

  const updateChatHistory = (newMessage, sender) => {
    setChatHistory((prevHistory) => {
      // 현재 selectedChatIndex가 없을 경우 기본값으로 설정
      let currentChatIndex = selectedChatIndex;
  
      if (currentChatIndex === null) {
        currentChatIndex = prevHistory.length > 0 ? prevHistory.length - 1 : 0;
        setSelectedChatIndex(currentChatIndex);
      }
  
      const chatExists = prevHistory.some(chat => chat.index === currentChatIndex);
  
      if (chatExists) {
        // 기존 채팅창에 메시지 추가
        const updatedHistory = prevHistory.map((chat) => {
          if (chat.index === currentChatIndex) {
            return { ...chat, messages: [...chat.messages, { sender, text: newMessage }] };
          }
          return chat;
        });
  
        console.log("Updated chat history (existing chat):", updatedHistory);
        return updatedHistory;
      } else {
        // 새로운 채팅창 생성
        const newChatTitle = new Date().toLocaleString();
        const newChat = {
          index: currentChatIndex,
          date: new Date().toLocaleDateString(),
          titles: [newChatTitle],
          messages: [{ sender, text: newMessage }],
        };
  
        const updatedHistory = [...prevHistory, newChat];
        console.log("Updated chat history (new chat):", updatedHistory);
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
    console.log("Selected index:", index);
    setSelectedChatIndex(index);
  
    // Recoil 상태에서 선택된 채팅을 찾고, 현재 채팅 상태를 업데이트
    setChatHistory((prevHistory) => {
      const selectedChat = prevHistory.find(chat => chat.index === index);
      if (selectedChat) {
        console.log("Selected chat:", selectedChat);
        setCurrentChat(selectedChat.messages || []);
      }
      return prevHistory; // Recoil 상태를 유지
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
      console.log("New chat added:", newHistory);
      return newHistory;
    });
    setCurrentChat([
      { sender: "server", text: "새로운 채팅이 시작되었습니다." },
      { sender: "server", text: "무엇을 도와드릴까요?" }
    ]);
    setSelectedChatIndex(newChat.index); // 새로운 타이틀 설정
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

  return (
    <RecoilRoot>
      <Box sx={{ position: "relative", width: "100vw", height: "100vh", bgcolor: "#1e1e1e" }}>
        {/* 사이드바 */}
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
            {/* 회사 로고 */}
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

            {/* 닫기 버튼 */}
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

            {/* 채팅 히스토리 토글 */}
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

            {/* 사이드바 리스트 */}
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
                              sx={{ bgcolor: "#f0f0f0", borderRadius: 3, flexGrow: 1 }} // 배경색을 회색 계열로 변경하고 모서리를 둥글게
                            />
                          ) : (
                            <>
                              <ListItemText primary={title} sx={{ color: "#fff", flexGrow: 1 }} />
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

        {/* 메인 화면 */}
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

          {/* 채팅 영역 */}
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
            {/* 로딩 스피너 */}
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

            {/* 채팅 메시지 */}
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

          {/* 입력 필드 */}
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
    </RecoilRoot>
  );
};

export default App;