import * as React from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import MuiCard from '@mui/material/Card';
import Checkbox from '@mui/material/Checkbox';
import FormLabel from '@mui/material/FormLabel';
import FormControl from '@mui/material/FormControl';
import FormControlLabel from '@mui/material/FormControlLabel';
import Link from '@mui/material/Link';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import { styled } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';
import { SitemarkIcon } from './CustomIcons';

const Card = styled(MuiCard)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignSelf: 'center',
  width: '100%',
  padding: theme.spacing(4),
  gap: theme.spacing(2),
  boxShadow:
    'hsla(220, 30%, 5%, 0.05) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.05) 0px 15px 35px -5px',
  [theme.breakpoints.up('sm')]: {
    width: '450px',
  },
  ...theme.applyStyles('dark', {
    boxShadow:
      'hsla(220, 30%, 5%, 0.5) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.08) 0px 15px 35px -5px',
  }),
}));

export default function SignInCard() {
  const navigate = useNavigate();
  const [loginIdError, setLoginIdError] = React.useState(false);
  const [loginIdErrorMessage, setLoginIdErrorMessage] = React.useState('');
  const [passwordError, setPasswordError] = React.useState(false);
  const [passwordErrorMessage, setPasswordErrorMessage] = React.useState('');
  const [rememberMe, setRememberMe] = React.useState(false);

  const handleSubmit = (event) => {
    event.preventDefault();
    if (validateInputs()) {
      const data = new FormData(event.currentTarget);
      const loginId = data.get('loginId');
      const password = data.get('password');

      if (loginId === 'bespin' && password === '1234') {
        if (rememberMe) {
          localStorage.setItem('loginId', loginId);
          localStorage.setItem('password', password);
        } else {
          localStorage.removeItem('loginId');
          localStorage.removeItem('password');
        }
        sessionStorage.setItem('isAuthenticated', 'true');
        navigate('/chat');
      } else {
        alert('Invalid login ID or password');
      }
    }
  };

  const validateInputs = () => {
    const loginId = document.getElementById('loginId');
    const password = document.getElementById('password');

    let isValid = true;

    if (!loginId.value) {
      setLoginIdError(true);
      setLoginIdErrorMessage('Please enter your login ID.');
      isValid = false;
    } else {
      setLoginIdError(false);
      setLoginIdErrorMessage('');
    }

    if (!password.value || password.value.length < 4) {
      setPasswordError(true);
      setPasswordErrorMessage('Password must be at least 4 characters long.');
      isValid = false;
    } else {
      setPasswordError(false);
      setPasswordErrorMessage('');
    }

    return isValid;
  };

  const handleRememberMeChange = (event) => {
    setRememberMe(event.target.checked);
  };

  React.useEffect(() => {
    const savedLoginId = localStorage.getItem('loginId');
    const savedPassword = localStorage.getItem('password');
    if (savedLoginId && savedPassword) {
      document.getElementById('loginId').value = savedLoginId;
      document.getElementById('password').value = savedPassword;
      setRememberMe(true);
    }
  }, []);

  return (
    <Card variant="outlined">
      <Box sx={{ display: { xs: 'flex', md: 'none' } }}>
        <SitemarkIcon />
      </Box>
      <Typography
        component="h1"
        variant="h4"
        sx={{ width: '100%', fontSize: 'clamp(1.6rem, 10vw, 1.5rem)' }}
      >
       BepsinGlobal - AI Native Team
      </Typography>
      <Typography
        component="h1"
        variant="h1"
        sx={{ width: '100%', fontSize: 'clamp(1.2rem, 10vw, 1.3rem)' }}
      >
      Sign In 
      </Typography>
      <Box
        component="form"
        onSubmit={handleSubmit}
        noValidate
        sx={{ display: 'flex', flexDirection: 'column', width: '100%', gap: 2 }}
      >
        <FormControl>
          <FormLabel htmlFor="loginId">Login ID</FormLabel>
          <TextField
            error={loginIdError}
            helperText={loginIdErrorMessage}
            id="loginId"
            type="text"
            name="loginId"
            placeholder="Enter your login ID"
            autoComplete="username"
            autoFocus
            required
            fullWidth
            variant="outlined"
            color={loginIdError ? 'error' : 'primary'}
          />
        </FormControl>
        <FormControl>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <FormLabel htmlFor="password">Password</FormLabel>
            <Link
              component="button"
              type="button"
              onClick={() => console.log('Forgot password clicked')}
              variant="body2"
              sx={{ alignSelf: 'baseline' }}
            >
              Forgot your password?
            </Link>
          </Box>
          <TextField
            error={passwordError}
            helperText={passwordErrorMessage}
            name="password"
            placeholder="••••"
            type="password"
            id="password"
            autoComplete="current-password"
            required
            fullWidth
            variant="outlined"
            color={passwordError ? 'error' : 'primary'}
          />
        </FormControl>
        <FormControlLabel
          control={
            <Checkbox
              checked={rememberMe}
              onChange={handleRememberMeChange}
              color="primary"
            />
          }
          label="Remember me"
        />
        <Button type="submit" fullWidth variant="contained" onClick={validateInputs}>
          Sign in
        </Button>
        {/* <Typography sx={{ textAlign: 'center' }}>
          Don&apos;t have an account?{' '}
          <span>
            <Link
              href="/material-ui/getting-started/templates/sign-in/"
              variant="body2"
              sx={{ alignSelf: 'center' }}
            >
              Sign up
            </Link>
          </span>
        </Typography> */}
      </Box>
    </Card>
  );
}
