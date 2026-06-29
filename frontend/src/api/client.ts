import axios from "axios";
import { API_BASE_URL } from "@/lib/constants";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,
});

// Request interceptor — attach any future auth headers here
api.interceptors.request.use((config) => config);

// Response interceptor — normalise errors
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message = err.response?.data?.detail ?? err.message;
    return Promise.reject(new Error(message));
  },
);
