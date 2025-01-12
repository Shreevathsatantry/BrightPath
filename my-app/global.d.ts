interface SpeechRecognition {
    start(): void
    stop(): void
    onresult: ((event: SpeechRecognitionEvent) => void) | null
    continuous: boolean
    interimResults: boolean
  }
  
  interface SpeechRecognitionEvent {
    results: {
      [index: number]: {
        [index: number]: {
          transcript: string
        }
      }
    }
  }
  
  interface Window {
    SpeechRecognition: typeof SpeechRecognition
    webkitSpeechRecognition: typeof SpeechRecognition
  }
  