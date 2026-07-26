import ModelRunner from './components/ModelRunner'
import { registeredAdapters } from './adapters/registry'

export default function App() {
  return <ModelRunner adapters={registeredAdapters} />
}
