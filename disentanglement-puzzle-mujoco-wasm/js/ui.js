function requireElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing required DOM element: #${id}`);
  }
  return element;
}

export const UI = Object.freeze({
  canvasWrap: requireElement('canvasWrap'),
  statusBadge: requireElement('statusBadge'),
  languageSelect: requireElement('languageSelect'),
  complexity: requireElement('complexity'),
  complexityValue: requireElement('complexityValue'),
  seedInput: requireElement('seedInput'),
  randomizeSeed: requireElement('randomizeSeed'),
  generateBtn: requireElement('generateBtn'),
  resetBtn: requireElement('resetBtn'),
  pauseBtn: requireElement('pauseBtn'),
  copyUrlBtn: requireElement('copyUrlBtn'),
  debugToggle: requireElement('debugToggle'),
  followToggle: requireElement('followToggle'),
  turboHintToggle: requireElement('turboHintToggle'),
  familyValue: requireElement('familyValue'),
  gapRatioValue: requireElement('gapRatioValue'),
  difficultyValue: requireElement('difficultyValue'),
  wireLengthValue: requireElement('wireLengthValue'),
  separationValue: requireElement('separationValue'),
  contactValue: requireElement('contactValue'),
  timerValue: requireElement('timerValue'),
  seedEcho: requireElement('seedEcho'),
  modeValue: requireElement('modeValue'),
  staticDiagText: requireElement('staticDiagText'),
  runtimeDiagText: requireElement('runtimeDiagText'),
  logText: requireElement('logText'),
  runDiagBtn: requireElement('runDiagBtn'),
  downloadDiagBtn: requireElement('downloadDiagBtn'),
  downloadSpecBtn: requireElement('downloadSpecBtn'),
  downloadMjcfBtn: requireElement('downloadMjcfBtn'),
  copyLogBtn: requireElement('copyLogBtn'),
  overlayMessage: requireElement('overlayMessage'),
});
