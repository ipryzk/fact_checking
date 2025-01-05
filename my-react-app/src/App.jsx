import { useState } from 'react'
import './App.css'

function App() {
  const [claim, setClaim] = useState("");
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
    console.log(claimQuery); 

    // Sending server data
    try {
      const response = await fetch("http://localhost:8000/submit_claim/", {
        method: "POST", // Use POST instead of GET
        headers: {
          "Content-Type": "application/json", 
        },
        body: JSON.stringify(claimQuery), 
      });
      if (!response.ok) {
        throw new Error(`Response Status: ${response.status}`);
      }
      const json = await response.json();
      console.log(json)
    } catch (error) {
      console.log(error.message)
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
    </>
  );
}

export default App;
