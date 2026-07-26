import ModelRunner from './components/ModelRunner'
import { registeredAdapters } from './adapters/magenta'

export default function App() {
  return <ModelRunner adapters={registeredAdapters} />
}
