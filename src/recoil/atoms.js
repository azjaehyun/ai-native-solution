import { atom } from 'recoil';

export const chatHistoryState = atom({
  key: 'chatHistoryState',
  default: [],
});