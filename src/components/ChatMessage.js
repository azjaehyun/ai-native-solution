// ChatMessage.js
import React from 'react';
import Box from '@mui/material/Box';
import Avatar from '@mui/material/Avatar';
import Typography from '@mui/material/Typography';
import SmartToyTwoToneIcon from '@mui/icons-material/SmartToyTwoTone';

const ChatMessage = ({ msg, isServer }) => {
  return isServer ? (
    <Box
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
  );
};

export default ChatMessage;