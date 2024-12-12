import React from "react";
import { Drawer, List, ListItem, ListItemText, Box, Typography, IconButton } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import AddIcon from "@mui/icons-material/Add";

const Sidebar = ({ isOpen, toggleSidebar, chatHistory, handleNewChat }) => (
  <Drawer
    variant="persistent"
    open={isOpen}
    sx={{
      "& .MuiDrawer-paper": {
        width: "250px",
        bgcolor: "#333",
        color: "#fff",
      },
    }}
  >
    <Box display="flex" justifyContent="space-between" alignItems="center" p={2}>
      {/* 사이드바 활성화/비활성화 버튼 */}
      <IconButton color="inherit" onClick={toggleSidebar}>
        <MenuIcon />
      </IconButton>
      <Typography variant="h6" sx={{ flexGrow: 1, textAlign: "center" }}>
        BespinGlobal
      </Typography>
      <IconButton color="inherit" onClick={handleNewChat}>
        <AddIcon />
      </IconButton>
    </Box>
    <List>
      {chatHistory.map((group, index) => (
        <Box key={index} mb={2}>
          <Typography variant="body2" color="#aaa" px={2}>
            {group.date}
          </Typography>
          {group.titles.map((title, idx) => (
            <ListItem button key={idx} sx={{ "&:hover": { bgcolor: "#444" } }}>
              <ListItemText primary={title} />
            </ListItem>
          ))}
        </Box>
      ))}
    </List>
  </Drawer>
);

export default Sidebar;
