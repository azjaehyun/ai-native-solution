import React from "react";
import { Box, TextField, IconButton } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";

const ChatInput = ({ inputText, setInputText, handleSend, isWaitingForServer }) => (
  <Box sx={{ p: 1 }}>
    <TextField
      variant="outlined"
      fullWidth
      value={inputText} // <--- 올바르게 props로 받은 값 사용
      onChange={(e) => setInputText(e.target.value)} // <--- setInputText props로 받은 함수 호출
      disabled={isWaitingForServer}
      placeholder="메시지를 입력하세요"
      InputProps={{
        endAdornment: (
          <IconButton onClick={handleSend} disabled={isWaitingForServer}>
            <SendIcon />
          </IconButton>
        ),
      }}
    />
  </Box>
);

export default ChatInput;
