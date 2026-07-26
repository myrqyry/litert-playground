import ModelRunner from './components/ModelRunner'
import { registeredAdapters } from './adapters/musiccoca'

export default function App() {
  return <ModelRunner adapters={registeredAdapters} />
}
