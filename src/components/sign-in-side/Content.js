import * as React from 'react';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import WarningAmber from '@mui/icons-material/WarningAmber';
import AttachFile from '@mui/icons-material/AttachFile';
import Dashboard from '@mui/icons-material/Dashboard';
import FindInPage from '@mui/icons-material/FindInPage';

// import { SitemarkIcon } from './CustomIcons';

const items = [
  {
    icon: <Dashboard sx={{ color: 'text.secondary' }} />,
    title: 'ChatBot UI',
    description:
      '기본 제공 UI (Basic)',
  },
  {
    icon: <AttachFile sx={{ color: 'text.secondary' }} />,
    title: '지식문서 지원',
    description:
      '지식문서 지원 파일 타입 : PDF,Doc,TXT,Excel,md,csv',
  },
  {
    icon: <FindInPage sx={{ color: 'text.secondary' }} />,
    title: 'RAG 구성',
    description:
      'S3, Opensearch를 활용하여 RAG를 구성합니다.',
  },
  {
    icon: <WarningAmber sx={{ color: 'text.secondary' }} />,
    title: '기타',
    description:
      'OCR / 데이터 전처리 /파인튜닝은 지원되지 않습니다. ',
  },
];

export default function Content() {
  return (
    <Stack
      sx={{ flexDirection: 'column', alignSelf: 'center', gap: 4, maxWidth: 450 }}
    >
      <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
        <Typography variant="h4" component="div">
          PoC Type A 소개
        </Typography>
      </Box>
      {items.map((item, index) => (
        <Stack key={index} direction="row" sx={{ gap: 2 }}>
          {item.icon}
          <div>
            <Typography gutterBottom sx={{ fontWeight: 'medium' }}>
              {item.title}
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              {item.description}
            </Typography>
          </div>
        </Stack>
      ))}
    </Stack>
  );
}
