Finding the best intrusion detection system on WUSTL EHMS 2020 Dataset for Internet of Medical Things (IoMT) Cybersecurity Research Data Set.

Link: https://www.cse.wustl.edu/~jain/ehms/index.html

Useful Links:
https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240
https://www.tensorflow.org/tutorials/images/cnn

Data is in file "data.csv" 

**Network Flow Headers:**

- **Dir**: Direction of the data flow. This can refer to whether data is inbound or outbound between source and destination.
- **Flgs**: Flags set in the network packets, often used in protocols like TCP to indicate specific packet states (e.g., SYN, ACK).
- **SrcAddr**: Source IP address of the device sending the data.
- **DstAddr**: Destination IP address of the device receiving the data.
- **Sport**: Source port number, which identifies the specific process on the source machine.
- **Dport**: Destination port number, identifying the specific process on the destination machine.
- **SrcBytes**: Number of bytes sent from the source device.
- **DstBytes**: Number of bytes received by the destination device.
- **SrcLoad**: Load or bandwidth usage from the source device.
- **DstLoad**: Load or bandwidth usage at the destination device.
- **SrcGap**: Time gap between consecutive packets sent from the source.
- **DstGap**: Time gap between consecutive packets received by the destination.

**Packet Interval and Jitter**

- **SIntPkt**: Interval between packets sent by the source (in milliseconds or seconds).
- **DIntPkt**: Interval between packets received by the destination.
- **SIntPktAct**: Actual interval of packets at the source, considering possible disruptions or attacks.
- **DIntPktAct**: Actual interval of packets at the destination.
- **SrcJitter**: Variation in packet intervals (jitter) for packets sent by the source.
- **DstJitter**: Variation in packet intervals (jitter) for packets received by the destination.

**Packet Size:**

- **sMaxPktSz**: Maximum packet size sent by the source.
- **dMaxPktSz**: Maximum packet size received by the destination.
- **sMinPktSz**: Minimum packet size sent by the source.
- **dMinPktSz**: Minimum packet size received by the destination.

**Network Session:**

- **Dur**: Duration of the session or data transfer.
- **Trans**: Number of transmissions that occurred during the session.
- **TotPkts**: Total number of packets transmitted in the session.
- **TotBytes**: Total number of bytes transmitted in the session.
- **Load**: The overall load on the network from this session.

**Packet Loss and Rate**

- **Loss**: Total number of packets lost during transmission.
- **pLoss**: Percentage of packets lost.
- **pSrcLoss**: Percentage of packets lost at the source.
- **pDstLoss**: Percentage of packets lost at the destination.
- **Rate**: Rate of data transmission, which could be in bits per second or packets per second.

**MAC Addresses and Packet Count:**

- **SrcMac**: MAC address of the source device.
- **DstMac**: MAC address of the destination device.
- **Packet_num**: Number assigned to each packet, helping in identifying sequence.

**Biometric Data:**

- **Temp**: Body temperature of the patient.
- **SpO2**: Oxygen saturation level in the patient’s blood.
- **Pulse_Rate**: Patient’s pulse rate.
- **SYS**: Systolic blood pressure of the patient.
- **DIA**: Diastolic blood pressure of the patient.
- **Heart_rate**: Overall heart rate of the patient.
- **Resp_Rate**: Patient’s respiratory rate.
- **ST**: ST segment from an ECG reading, which can indicate heart issues.

**Attack Detection:**

- **Attack Category**: Type or category of attack detected (e.g., DoS, probe, or other intrusion types).
- **Label**: Label indicating whether the traffic is normal or part of an attack.