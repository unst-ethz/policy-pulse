import React, { Component, Suspense, lazy, type ReactNode } from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Link, NavLink, Outlet, Route, Routes } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import '@fontsource-variable/ibm-plex-sans/wght.css';
import { ErrorPanel, Loading } from './components/ui';
import './styles.css';

const Home = lazy(() => import('./pages/Home'));
const Explore = lazy(() => import('./pages/Explore'));
const Profile = lazy(() => import('./pages/Profile'));
const Resolution = lazy(() => import('./pages/Resolution'));
const Methodology = lazy(() => import('./pages/Methodology'));
const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    return this.state.error ? (
      <ErrorPanel
        error={new Error('The page could not be displayed. Reload to try again.')}
        retry={() => window.location.reload()}
      />
    ) : (
      this.props.children
    );
  }
}
function Layout() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header">
        <div className="header-inner">
          <Link className="brand" to="/">
            UN-ETH Policy Pulse
          </Link>
          <nav aria-label="Main navigation">
            <NavLink to="/" end>
              Overview
            </NavLink>
            <NavLink to="/trends">Explore data</NavLink>
            <NavLink to="/methodology">Methodology</NavLink>
          </nav>
          <a
            className="team-link"
            href="https://github.com/unst-ethz/policy-pulse"
            target="_blank"
            rel="noreferrer"
          >
            Open source <ArrowUpRight size={14} />
          </a>
        </div>
      </header>
      <main id="main">
        <ErrorBoundary>
          <Suspense fallback={<Loading />}>
            <Outlet />
          </Suspense>
        </ErrorBoundary>
      </main>
      <footer>
        <div>
          <strong>UN-ETH Policy Pulse</strong>
          <p>
            A volunteer project of the United Nations Student Team at ETH Zürich,
            <br />
            in collaboration with the UN Dag Hammarskjöld Library.
          </p>
        </div>
        <div className="footer-links">
          <a href="https://ethz.ch">ETH Zürich</a>
          <a href="https://un-eth.ethz.ch/exchanges/un-eth-student-team.html">
            UN-ETH Student Team
          </a>
          <a href="https://www.un.org/en/library">Dag Hammarskjöld Library</a>
          <Link to="/methodology">Sources &amp; limitations</Link>
        </div>
      </footer>
    </>
  );
}
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="trends" element={<Explore />} />
            <Route path="explore" element={<Explore />} />
            <Route path="profile" element={<Profile />} />
            <Route path="resolutions/:id" element={<Resolution />} />
            <Route path="methodology" element={<Methodology />} />
            <Route
              path="*"
              element={
                <div className="state">
                  <h1>Page not found</h1>
                  <Link to="/trends">Explore voting data</Link>
                </div>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
