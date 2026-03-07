import { PuzzleApp } from './app.js';
import { detectInitialLocale } from './i18n.js';

const app = new PuzzleApp({ locale: detectInitialLocale() });
app.init();
