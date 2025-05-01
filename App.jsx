
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Mail, Linkedin } from "lucide-react"

export default function AmirPortfolio() {
  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="max-w-4xl mx-auto">
        <img src="/Amir.jpeg" alt="Amir Exir" className="rounded-full w-40 h-40 object-cover mx-auto mb-6" />
        <h1 className="text-4xl font-bold text-center mb-4">Amir Exir, M.Eng EE, P.E., NCSO</h1>
        <p className="text-center text-gray-600 mb-8">
          Power Systems Engineer | AI Researcher | EMS & Planning Expert | UT Austin AI Graduate Student
        </p>

        <div className="flex justify-center gap-4 mb-12">
          <a href="mailto:eksir.monfared.amir@gmail.com">
            <Button variant="outline"><Mail className="mr-2" />Email</Button>
          </a>
          <a href="https://www.linkedin.com/in/amir-exir-m-eng-ee-p-e-ncso-6323b3153/" target="_blank">
            <Button variant="outline"><Linkedin className="mr-2" />LinkedIn</Button>
          </a>
        </div>

        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-4">Professional Summary</h2>
          <p className="text-gray-700">
            Professional Engineer (P.E.) and NERC Certified Reliability Coordinator with 6+ years in power systems across operation, planning,
            and modeling at ERCOT and LCRA. Experienced in PSS/E, GE EMS SCADA/TSM/DTS, and Python. Currently pursuing an M.S. in AI at UT Austin.
          </p>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-4">Education</h2>
          <ul className="list-disc list-inside text-gray-700">
            <li><strong>University of Texas at Austin</strong> – M.S. in Artificial Intelligence (4.0 GPA), 2024–Present</li>
            <li><strong>Lamar University</strong> – M.Eng. in Electrical & Computer Engineering, 2020</li>
            <li><strong>Shahid Beheshti University</strong> – B.Sc. in Electrical Engineering, 2017</li>
          </ul>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-4">Licenses & Certifications</h2>
          <ul className="list-disc list-inside text-gray-700">
            <li>P.E. License – Texas Board of Professional Engineers (#151267)</li>
            <li>NERC System Operator RC Certification</li>
            <li>IBM ML & Python Certifications</li>
            <li>Canvas Badges: Deep Learning, Ethics in AI (UT Austin)</li>
          </ul>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-4">Work Experience</h2>
          <ul className="space-y-4 text-gray-700">
            <li>
              <strong>Transmission Planning Model Engineer – LCRA (2024–Present)</strong>
              <p>Perform steady-state and dynamic analysis for planning & capital projects. Ensure ERCOT compliance. Submit model changes via PMCR/DCP.</p>
            </li>
            <li>
              <strong>EMS & Operation Network Model Engineer – LCRA (2022–2024)</strong>
              <p>Managed EMS/SCADA models, one-line diagrams, and DTS systems. Submitted NOMCR to ERCOT. Collaborated with regional co-ops.</p>
            </li>
            <li>
              <strong>Real-Time Shift Engineer – ERCOT (2022)</strong>
              <p>Supported control room operators via power flow & stability assessments. Managed contingency analysis tools and situational awareness systems.</p>
            </li>
            <li>
              <strong>Operations Instructor – ERCOT (2020–2022)</strong>
              <p>Developed operator training and simulator scenarios for ERCOT staff including black start, RTA, and IROL emergencies.</p>
            </li>
          </ul>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-4">Projects</h2>
          <ul className="list-disc list-inside text-gray-700">
            <li>Developed AI racing agent using deep learning (UT Austin, SuperTuxKart project)</li>
            <li>Built tools for generation/load interconnection planning and congestion analysis</li>
            <li>Conducted ethical AI assessments and fairness analysis in model deployment</li>
          </ul>
        </section>

        <section className="mb-10">
          <h2 className="text-2xl font-semibold mb-4">In Action</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <img src="/Control Room.jpeg" alt="ERCOT Control Room" className="rounded-lg shadow" />
            <img src="/0084F2C3-6631-494C-B1BE-2E1948A72E9C.jpeg" alt="Amir in Substation" className="rounded-lg shadow" />
          </div>
        </section>

        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-4">Download My Resume & Certs</h2>
          <div className="flex gap-4 flex-wrap">
            <a href="/CV8.docx" download><Button variant="outline">📄 Resume</Button></a>
            <a href="/Coursera BYBV066EFBS7.pdf" download><Button variant="outline">📜 Python Certificate</Button></a>
            <a href="/Coursera DCHGTCGASDAH.pdf" download><Button variant="outline">📜 ML Certificate</Button></a>
            <a href="/Deep Learning - Canvas Badges.pdf" download><Button variant="outline">🧠 Deep Learning</Button></a>
            <a href="/Ethics in AI - Canvas Badges.pdf" download><Button variant="outline">⚖️ Ethics in AI</Button></a>
          </div>
        </section>
      </div>
    </div>
  )
}
