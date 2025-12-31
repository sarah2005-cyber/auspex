import { createTheme } from '@mui/material/styles'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#6a1b9a' // deep purple
    },
    secondary: {
      main: '#ab47bc'
    },
    background: {
      default: '#f3e8ff'
    }
  },
  typography: {
    fontFamily: 'Inter, Roboto, Arial'
  }
})

export default theme
