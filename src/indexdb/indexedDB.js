const DB_NAME = 'chatDB';
const DB_VERSION = 1;
const STORE_NAME = 'chatHistory';

const openDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'index' });
      }
    };

    request.onsuccess = (event) => {
      resolve(event.target.result);
    };

    request.onerror = (event) => {
      reject(event.target.error);
    };
  });
};

export const getChatHistory = async () => {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onsuccess = (event) => {
      const result = event.target.result;
      // 인덱스를 다시 설정
      result.forEach((chat, index) => {
        chat.index = index;
      });
      resolve(result);
    };

    request.onerror = (event) => {
      reject(event.target.error);
    };
  });
};

export const saveChatHistory = async (chatHistory) => {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    // 인덱스를 다시 설정
    chatHistory.forEach((chat, index) => {
      chat.index = index;
      store.put(chat);
    });

    transaction.oncomplete = () => {
      resolve();
    };

    transaction.onerror = (event) => {
      reject(event.target.error);
    };
  });
};