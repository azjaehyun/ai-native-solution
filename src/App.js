import React, { useState, useRef, useEffect } from "react";
import {
  AppBar,
  Button,
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
  useMediaQuery,
  useTheme
} from "@mui/material";
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import MenuIcon from "@mui/icons-material/Menu";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import SendIcon from "@mui/icons-material/Send";
import NoteAddIcon from "@mui/icons-material/LocalHospital";
import SmartToyTwoToneIcon from "@mui/icons-material/SmartToyTwoTone";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import { getChatHistory, saveChatHistory } from './indexdb/indexedDB';
import LinearProgress from '@mui/material/LinearProgress';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import CancelIcon from '@mui/icons-material/Cancel';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const App = () => {

  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState("claude3.5");
  const [currentChat, setCurrentChat] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isWaitingForServer, setIsWaitingForServer] = useState(false);
  const [isAlertOpen, setAlertOpen] = useState(false);
  const [isChatHistoryEnabled, setIsChatHistoryEnabled] = useState(true);
  const [isEditing, setIsEditing] = useState(null);
  const [newTitle, setNewTitle] = useState("");
  const [selectedChatIndex, setSelectedChatIndex] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [fileLimitAlert, setFileLimitAlert] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const controllerRef = useRef(null);
  const chatBoxRef = useRef(null);

  const chatHistoryLimit = 10;

  useEffect(() => {
    const loadChatHistory = async () => {
      const history = await getChatHistory();
      setChatHistory(history);
    };
    loadChatHistory();
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [currentChat]);



  const sendMessageToServer = async (message,chatHistory,model,file) => {
    try {
      console.log("message", message);
      const chatHistoryMessage = chatHistory ? chatHistory.messages : [];
      console.log("chatHistoryMessage", chatHistoryMessage);

      console.log("file : ",file);

      let fileContent = null;
      let fileName = null;
      if (file) {
        fileContent = await getBase64(file);
        fileName = file.name;
      }

      console.log("getBase64 file : ",file);
      const response = await axios.post('https://x4v4n6sd92.execute-api.ap-northeast-2.amazonaws.com/prd/poc-type-a',
         { message , chatHistoryMessage , model , fileContent , fileName }
      );
      const responseData = response.data;
      console.log("responseData", responseData.resultData.message);
      return responseData.resultData.message.replace(/\n/g, '<br/>');
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

  
    const serverMessage = await sendMessageToServer(newUserMessage, chatHistory[selectedChatIndex], selectedModel, uploadedFile);
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
  
    // 파일 전송 후 파일 상태 초기화
    setUploadedFile(null);
    setUploadProgress(0);
  };


  const getBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        try {
          const base64String = reader.result.split(',')[1];
          console.log("base64String : ",base64String)
          resolve(base64String);
        } catch (error) {
          console.error('Error processing file:', error);
          reject('Error processing file');
        }
      };
      reader.onerror = (error) => {
        console.error('Error reading file:', error);
        reject('Error reading file');
      };
    });
  };




  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    console.log("handleFileUpload.file", file);

    if (file) {
      const fileSizeInMB = file.size / (1024 * 1024); // Convert bytes to MB
      if (fileSizeInMB > 10) {
        setFileLimitAlert(true); // Show alert if file is too large
        return;
      }

      setUploadedFile(file);
      setUploadProgress(0);

      // Simulate file upload progress
      const interval = setInterval(() => {
        setUploadProgress((prevProgress) => {
          if (prevProgress >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prevProgress + 10;
        });
      }, 200);
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

  const handleCancelUpload = () => {
    setUploadedFile(null);
    setUploadProgress(0);
  };

  const handleCloseAlert = () => {
    setFileLimitAlert(false);
  };

  const [logoutDialogOpen, setLogoutDialogOpen] = useState(false);
  const navigate = useNavigate();

  // const handleLogoutClick = () => {
  //   setLogoutDialogOpen(true);
  // };

  const handleLogoutConfirm = () => {
    sessionStorage.setItem('isAuthenticated', 'false');
    setLogoutDialogOpen(false);
    navigate('/');
  };

  const handleLogoutCancel = () => {
    setLogoutDialogOpen(false);
  };

  const sortedChatHistory = [...chatHistory].sort((a, b) => new Date(b.date) - new Date(a.date));

  const groupedChatHistory = sortedChatHistory.reduce((acc, chat, index) => {
    const date = chat.date;
    if (!acc[date]) {
      acc[date] = [];
    }
    acc[date].push({ title: chat.titles[0], index: chat.index });
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
                    <ListItem key={titleIndex} button="true" onClick={() => handleChatSelect(title.index)}>
                      <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        {isEditing === title.index ? (
                          <TextField
                            value={newTitle}
                            onChange={(e) => setNewTitle(e.target.value)}
                            onBlur={() => handleSaveTitle(title.index)}
                            onKeyPress={(e) => {
                              if (e.key === "Enter") {
                                handleSaveTitle(title.index);
                              }
                            }}
                            onClick={(event) => event.stopPropagation()}
                            autoFocus
                            sx={{ bgcolor: "#f0f0f0", borderRadius: 3, flexGrow: 1 }}
                          />
                        ) : (
                          <>
                            <ListItemText primary={title.title} sx={{ color: title.index === selectedChatIndex ? "#a8dadc" : "#fff", flexGrow: 1 }} />
                            <IconButton onClick={(event) => { event.stopPropagation(); handleEditTitle(title.index, event); }}>
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
            <Typography variant="h6"
                  sx={{
                    flexGrow: 1,
                    fontSize: isSmallScreen ? '1rem' : '1.5rem', // 작은 화면에서는 작은 글자 크기, 큰 화면에서는 큰 글자 크기
                  }} 
            >
              AWS Bedrock PoC
            </Typography>
            <Select
              value={selectedModel}
              onChange={handleModelChange}
              sx={{ bgcolor: "#444", color: "#fff", borderRadius: 1 }}
            >
              <MenuItem value="claude3.5">Claude3.5</MenuItem>
              <MenuItem value="claude3.0">Claude3.0</MenuItem>
            </Select>
           
              <Dialog
                open={logoutDialogOpen}
                onClose={handleLogoutCancel}
                aria-labelledby="logout-dialog-title"
                aria-describedby="logout-dialog-description"
              >
                <DialogTitle id="logout-dialog-title">{"Logout Confirmation"}</DialogTitle>
                <DialogContent>
                  <DialogContentText id="logout-dialog-description">
                    Are you sure you want to log out?
                  </DialogContentText>
                </DialogContent>
                <DialogActions>
                  <Button onClick={handleLogoutCancel} color="primary">
                    Cancel
                  </Button>
                  <Button onClick={handleLogoutConfirm} color="primary" autoFocus>
                    Confirm
                  </Button>
                </DialogActions>
            </Dialog>
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
                position: "fixed",
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
                
                dangerouslySetInnerHTML={{ __html: msg.text }}
                />
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

       
        <Box>
    
        {uploadProgress > 0 && uploadProgress < 100 && (
          <LinearProgress variant="determinate" value={uploadProgress} sx={{ mt: 1, width: '100%' }} />
        )}
        </Box>
        <Snackbar
          open={fileLimitAlert}
          autoHideDuration={3000}
          onClose={handleCloseAlert}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
          <Alert onClose={handleCloseAlert} severity="error" sx={{ width: '100%' }}>
            파일은 10MB 이하만 가능합니다.
          </Alert>
        </Snackbar>
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
                    <Tooltip title={uploadedFile ? uploadedFile.name : "파일 업로드"}>
                      <IconButton component="span" sx={{ ml: 1 }}>
                        <UploadFileIcon />
                      </IconButton>
                    </Tooltip>
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
              startAdornment: uploadedFile && (
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <Tooltip title={uploadedFile.name}>
                    <AttachFileIcon sx={{ mr: 0.5 }} />
                  </Tooltip>
                  <IconButton onClick={handleCancelUpload} sx={{ ml: 0 }}>
                    <CancelIcon />
                  </IconButton>
                </div>
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