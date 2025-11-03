const express = require('express');
const nodemailer = require('nodemailer');
const cors = require('cors');
const app = express();

const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.GMAIL_USER || 'pumazeta@gmail.com',
    pass: process.env.GMAIL_APP_PASSWORD || 'tua-app-password'
  }
});

app.post('/send-report', async (req, res) => {
  try {
    const { activities, patientInfo } = req.body;
    
    console.log('📨 Ricevuta richiesta per', activities.length, 'attività');

    // Genera il file TXT
    let fileContent = `PAZIENTE: ${patientInfo.name}\n`;
    fileContent += `EMAIL: ${patientInfo.email}\n`;
    fileContent += `DATA_INVIO: ${new Date().toLocaleString('it-IT')}\n`;
    fileContent += `TOTALE_ATTIVITA: ${activities.length}\n\n`;

    activities.forEach((activity, index) => {
      fileContent += `ATTIVITA_${index + 1}\n`;
      fileContent += `TYPE=${activity.tipo}\n`;
      fileContent += `NAME=${activity.nome}\n`;
      fileContent += `INTENSITY=${activity.intensita}\n`;
      fileContent += `DURATION=${activity.durata}min\n`;
      fileContent += `STARTTIME=${activity.data_ora}\n`;
      if (activity.cibi) fileContent += `FOODS=${activity.cibi}\n`;
      if (activity.note) fileContent += `NOTES=${activity.note}\n`;
      fileContent += `TIMESTAMP=${activity.timestamp}\n\n`;
    });

    const mailOptions = {
      from: 'App Monitoraggio <pumazeta@gmail.com>',
      to: 'robertocolucci@libero.it',
      subject: `Report Attività - ${new Date().toLocaleDateString('it-IT')}`,
      text: `Report automatico con ${activities.length} attività registrate.\n\nIl file dettagliato è in allegato.`,
      attachments: [{
        filename: `attivita_${Date.now()}.txt`,
        content: fileContent
      }]
    };

    const info = await transporter.sendMail(mailOptions);
    console.log('✅ Email INVIATA! Message ID:', info.messageId);
    
    res.json({ success: true, message: 'Email inviata con allegato' });
    
  } catch (error) {
    console.error('❌ Errore invio email:', error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Route di test AGGIORNATA
app.get('/test', (req, res) => {
  res.json({ message: 'Server funzionante! 🚀' });
});

app.get('/', (req, res) => {
  res.json({ message: 'Server Email Monitoraggio Cardiaco' });
});

app.listen(PORT, () => {
  console.log('🚀 Server email in esecuzione sulla porta', PORT);
});