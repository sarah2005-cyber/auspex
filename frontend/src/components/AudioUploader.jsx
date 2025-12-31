import React, {useState} from 'react'
import { Box, Button, Typography, LinearProgress, Stack, Card, CardContent } from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import axios from 'axios'

export default function AudioUploader(){
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [shapUrl, setShapUrl] = useState(null)

  const onFileChange = (e) => {
    setFile(e.target.files[0])
    setResult(null)
    setShapUrl(null)
  }

  const postFile = async (endpoint) => {
    if(!file) return
    setUploading(true)
    setResult(null)
    setShapUrl(null)
    try{
      const fd = new FormData()
      fd.append('file', file)
      const res = await axios.post(endpoint, fd, { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120000 })
      if(res.data){
        setResult(res.data)
        if(res.data.plot_url){
          // backend gives /static path
          setShapUrl(res.data.plot_url)
        } else {
          // fallback to known static filename
          setShapUrl('/static/test_shap.png')
        }
      }
    }catch(err){
      console.error(err)
      setResult({ error: err.message || 'Upload failed' })
    }finally{
      setUploading(false)
    }
  }

  return (
    <Box>
      <Stack spacing={2} direction="column">
        <input accept="audio/*" id="audio-file" type="file" style={{ display: 'none' }} onChange={onFileChange} />
        <label htmlFor="audio-file">
          <Button variant="contained" color="secondary" component="span" startIcon={<UploadFileIcon/>}>
            Choose audio file
          </Button>
        </label>
        {file && <Typography variant="body1">Selected: {file.name} ({Math.round(file.size/1024)} KB)</Typography>}

        <Stack direction="row" spacing={2}>
          <Button variant="contained" onClick={() => postFile('http://127.0.0.1:8000/predict_file')} disabled={!file || uploading}>
            Predict
          </Button>
          <Button variant="outlined" onClick={() => postFile('http://127.0.0.1:8000/explain_file')} disabled={!file || uploading}>
            Explain (SHAP)
          </Button>
        </Stack>

        {uploading && <LinearProgress />}

        {result && (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1">Result</Typography>
              <pre style={{whiteSpace: 'pre-wrap'}}>{JSON.stringify(result, null, 2)}</pre>
            </CardContent>
          </Card>
        )}

        {shapUrl && (
          <Box>
            <Typography variant="subtitle1">SHAP Plot</Typography>
            <img src={shapUrl} alt="SHAP" style={{ maxWidth: '100%', borderRadius: 6, marginTop: 8 }} />
          </Box>
        )}
      </Stack>
    </Box>
  )
}
