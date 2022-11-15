// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getStorage, ref } from "firebase/storage";


// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBhpkLtMoCmAq-wiztiRkqZ3IMzzv1YEfE",
  authDomain: "stock-price-ee1ca.firebaseapp.com",
  databaseURL: "https://stock-price-ee1ca-default-rtdb.firebaseio.com",
  projectId: "stock-price-ee1ca",
  storageBucket: "stock-price-ee1ca.appspot.com",
  messagingSenderId: "874834751520",
  appId: "1:874834751520:web:e7c2adaddd0e2d03938c45",
  measurementId: "G-3CP8R3JM9H"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

const storage = getStorage();
const pathReference = ref(storage, 'uploads/');