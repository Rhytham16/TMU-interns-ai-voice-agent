
    let isLogin = true;

    const form = document.getElementById("authForm");
    const message = document.getElementById("message");

    const registerUsernameGroup = document.getElementById("register-username-group");
    const registerEmailGroup = document.getElementById("register-email-group");
    const toggleText = document.getElementById("toggleText");
    const formTitle = document.getElementById("form-title");
    const submitBtn = document.getElementById("submitBtn");

    function toggleForm() {
      isLogin = !isLogin;

      if (isLogin) {
        formTitle.textContent = "Login";
        submitBtn.textContent = "Login";
        registerUsernameGroup.style.display = "none";
        registerEmailGroup.style.display = "none";
        toggleText.textContent = "Register here";
      } else {
        formTitle.textContent = "Register";
        submitBtn.textContent = "Register";
        registerUsernameGroup.style.display = "block";
        registerEmailGroup.style.display = "block";
        toggleText.textContent = "Login here";
      }
      document.getElementById("usernameOrEmail").disabled = !isLogin;
      message.textContent = "";
      message.className = "message";
      form.reset();
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      console.log("Form submitted. isLogin:", isLogin); // DEBUG

      message.textContent = "Processing...";
      message.className = "message loading";

      try {
        if (isLogin) {
          const login_id = document.getElementById("usernameOrEmail").value.trim();
          const password = document.getElementById("password").value.trim();

          const res = await fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ login_id, password })
          });

          const result = await res.json();

          if (result.message) {
            message.textContent = result.message;
            message.className = "message success";
            setTimeout(() => {
              window.location.href = "/static/chat.html";
            }, 1000);
          } else {
            throw new Error(result.error || "Login failed");
          }
        } else {
          const username = document.getElementById("registerUsername").value.trim();
          const email = document.getElementById("registerEmail").value.trim();
          const password = document.getElementById("password").value.trim();

          if (!username || !email || !password) {
            throw new Error("Please fill in all registration fields.");
          }

          const res = await fetch("/signup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, email, password })
          });

          const result = await res.json();
          console.log("Signup response:", result); // DEBUG

          if (result.message) {
            message.textContent = result.message;
            message.className = "message success";
            setTimeout(() => {
              toggleForm();
            }, 1500);
          } else {
            throw new Error(result.error || "Registration failed");
          }
        }
      } catch (err) {
        console.error("Error:", err); // DEBUG
        message.textContent = err.message;
        message.className = "message error";
      }
    });
