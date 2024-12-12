import React from "react";
import { Box, Avatar, Typography, TextField, IconButton, CircularProgress } from "@mui/material";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import SendIcon from "@mui/icons-material/Send";
import StopIcon from "@mui/icons-material/Stop";

const ChatBox = ({
  chatBoxRef,
  currentChat,
  inputText,
  setInputText,
  handleSend,
  handleStopRequest,
  isWaitingForServer,
}) => (
  <Box
    sx={{
      display: "flex",
      flexDirection: "column",
      height: "100%",
    }}
  >
    {/* 채팅 메시지 영역 */}
    <Box ref={chatBoxRef} sx={{ flexGrow: 1, p: 2, overflowY: "auto", bgcolor: "#1e1e1e" }}>
      {currentChat.map((msg, idx) => (
        <Box
          key={idx}
          display="flex"
          alignItems="center"
          justifyContent={msg.sender === "user" ? "flex-end" : "flex-start"}
          my={1}
        >
          {msg.sender === "server" && (
            <Avatar sx={{ bgcolor: "#1976d2", mr: 1 }}>
              <SmartToyIcon />
            </Avatar>
          )}
          <Typography
            sx={{
              bgcolor: msg.sender === "user" ? "#1976d2" : "#444",
              color: "#fff",
              p: 1,
              borderRadius: 2,
              maxWidth: "70%",
            }}
          >
            {msg.text}
          </Typography>
        </Box>
      ))}
    </Box>

    {/* 입력 필드 및 중지 버튼 */}
    <Box sx={{ display: "flex", alignItems: "center", p: 1, bgcolor: "#f5f5f5" }}>
      <TextField
        variant="outlined"
        fullWidth
        disabled={isWaitingForServer} // 서버 응답 대기 중에는 입력 비활성화
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        onKeyPress={(e) => {
          if (e.key === "Enter" && !isWaitingForServer) {
            e.preventDefault();
            handleSend();
          }
        }}
        placeholder="메시지를 입력하세요..."
        sx={{
          bgcolor: "#fff",
          borderRadius: 1,
        }}
        InputProps={{
          endAdornment: (
            <IconButton onClick={handleSend} disabled={isWaitingForServer}>
              <SendIcon />
            </IconButton>
          ),
        }}
      />
      {isWaitingForServer && (
        <IconButton onClick={handleStopRequest} sx={{ ml: 1 }}>
          <StopIcon />
        </IconButton>
      )}
    </Box>

    {/* 로딩 스피너 */}
    {isWaitingForServer && (
      <Box
        sx={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          bgcolor: "rgba(0, 0, 0, 0.6)",
          borderRadius: 2,
          p: 2,
          zIndex: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <CircularProgress color="info" />
        <Typography sx={{ color: "#fff", ml: 2 }}>서버 응답을 기다리는 중...</Typography>
      </Box>
    )}
  </Box>
);

export default ChatBox;
