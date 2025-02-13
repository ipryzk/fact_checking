import { useState } from 'react'
import './App.css'

function App() {
  const [claim, setClaim] = useState("");
  const [jsonData, setJsonData] = useState({ corroborate: [], contradict: [] });

  const [checkboxes, setCheckboxes] = useState({
    "journal_article": false,
    "book_chapter": false,
    "proceedings_article": false,
    "report": false,
    "standard": false,
    "dataset": false,
    "posted_content": false,
    "dissertation": false,
  });
  

  // Handle checkbox change
  const handleCheckboxChange = (event) => {
    const { name, checked } = event.target;
    setCheckboxes((prevCheckboxes) => ({
      ...prevCheckboxes,
      [name]: checked,
    }));
  };

  // Handle "Select All" click
  const handleSelectAll = () => {
    const updatedCheckboxes = Object.keys(checkboxes).reduce((acc, key) => {
      acc[key] = true;
      return acc;
    }, {});
    setCheckboxes(updatedCheckboxes);
  };

  // Handle form submission
  const handleSubmit = async () => {
    const claimQuery = {
      claim: claim,
      ...checkboxes,
    };
  
    try {
      const response = await fetch("http://localhost:8000/submit_claim/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(claimQuery),
      });
  
      if (!response.ok) {
        throw new Error(`Response Status: ${response.status}`);
      }
  
      const json = await response.json();
      console.log(json); // Inspect response
  
      // Check if the response is an object with expected keys
      if (json && typeof json === "object" && "corroborate" in json && "contradict" in json) {
        setJsonData(json);  // Store the object directly in state
      } else {
        setJsonData({ corroborate: [], contradict: [] });  // Ensure the structure is always correct
      }
    } catch (error) {
      console.log(error.message);
    }
  };
  
  

  return (
    <>
      <div>
        <h1>Welcome!</h1>
        <h2>Please fill in the following fields below:</h2>
      </div>
      <div>
        <p>Your claim:</p>
        <input
          type="text"
          id="claim"
          value={claim}
          onChange={(e) => setClaim(e.target.value)} // Update claim state
        />
        <p>Filter by:</p>
        <form>
          <label>
            <input
              type="button"
              id="selectAll"
              value="Select All"
              onClick={handleSelectAll} // Trigger Select All
            />
          </label> <br /><br />
          {Object.keys(checkboxes).map((key) => (
            <label key={key}>
              <input
                type="checkbox"
                name={key}
                checked={checkboxes[key]} // Controlled checkbox
                onChange={handleCheckboxChange} // Handle checkbox change
              />
              {key}
            </label> 
          ))}
        </form>
        <button id="submission" onClick={handleSubmit}>Submit</button>
      </div>
      {jsonData && jsonData.corroborate && jsonData.contradict && (
  <>
    <h2>Corroborate</h2>
    {jsonData.corroborate.length > 0 ? (
      jsonData.corroborate.map((item, index) => (
        <div key={`corroborate-${index}`}>
          <h3>CLAIM: </h3>
          <p>{item.claim}</p>
          <h3>TEXT: </h3>
          <p>{item.text}</p>
          <h3>JUSTIFICATION: </h3>
          <p>{item.justification}</p>
          <h3>DOI: </h3>
          <p>{item.doi}</p>
        </div>
      ))
    ) : (
      <p>No corroborating evidence found.</p>
    )}

    <h2>Contradict</h2>
    {jsonData.contradict.length > 0 ? (
      jsonData.contradict.map((item, index) => (
        <div key={`contradict-${index}`}>
          <h3>CLAIM: </h3>
          <p>{item.claim}</p>
          <h3>TEXT: </h3>
          <p>{item.text}</p>
          <h3>JUSTIFICATION: </h3>
          <p>{item.justification}</p>
          <h3>DOI: </h3>
          <p>{item.doi}</p>
        </div>
      ))
    ) : (
      <p>No contradicting evidence found.</p>
    )}
  </>
)}


    </>
  );
}

export default App;
